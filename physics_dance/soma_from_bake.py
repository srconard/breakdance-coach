"""Bake npz (skeleton_bake_npz.py: JW, G, rest; SMPL joint indices, Y-up)
-> ProtoMotions SOMA23 .motion file.

Runs INSIDE the ProtoMotions checkout (cwd = repo root), in the Modal image.

The bake's G is each source bone's world rotation relative to its own rest
(T-pose) frame, so it transfers directly onto SOMA23, whose MJCF zero pose is
also a T-pose with identity body frames. Only the world basis changes:
source Y-up / faces +Z / left +X  ->  SOMA Z-up / faces -Y / left +X.

    python soma_from_bake.py in.npz out_dir/name.motion [--start s] [--end s]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "data/scripts")
from contact_detection import compute_contact_labels_from_pos_and_vel  # noqa: E402
from protomotions.components.pose_lib import (  # noqa: E402
    compute_angular_velocity,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    fk_from_transforms_with_velocities,
)
from protomotions.utils.rotations import matrix_to_quaternion  # noqa: E402

MJCF = "protomotions/data/assets/mjcf/soma23_humanoid.xml"
MJCF_BODY_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
]
# SOMA23 body -> SMPL joint index in the bake
SMPL_OF = {
    "Hips": 0, "Spine1": 3, "Spine2": 6, "Chest": 9, "Neck1": 12, "Neck2": 12,
    "Head": 15, "RightShoulder": 14, "RightArm": 17, "RightForeArm": 19,
    "RightHand": 21, "LeftShoulder": 13, "LeftArm": 16, "LeftForeArm": 18,
    "LeftHand": 20, "RightLeg": 2, "RightShin": 5, "RightFoot": 8,
    "RightToeBase": 11, "LeftLeg": 1, "LeftShin": 4, "LeftFoot": 7,
    "LeftToeBase": 10,
}
# Y-up/+Z-forward -> Z-up/-Y-forward (a proper rotation, Rx(+90 deg))
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
# clearance of each body's origin above the floor when that body touches it
CLEAR = {"RightFoot": 0.075, "LeftFoot": 0.075, "RightToeBase": 0.03,
         "LeftToeBase": 0.03, "RightHand": 0.035, "LeftHand": 0.035,
         "RightShin": 0.06, "LeftShin": 0.06, "Hips": 0.10, "Head": 0.10}
SOMA_HIP_TO_ANKLE = 0.938  # Hips->Leg->Shin->Foot rest drop in soma23_humanoid.xml


def retime(JW, G, speed):
    """Play the clip at `speed` (0.5 = half speed) by resampling: joint
    positions linearly, rotations by per-joint slerp."""
    from scipy.spatial.transform import Rotation, Slerp

    F = len(JW)
    t_src = np.arange(F)
    t_new = np.arange(0, F - 1 + 1e-9, speed)
    JWn = np.stack([np.stack([np.interp(t_new, t_src, JW[:, j, k]) for k in range(3)], -1)
                    for j in range(JW.shape[1])], 1)
    Gn = np.zeros((len(t_new), G.shape[1], 3, 3))
    for j in range(G.shape[1]):
        Gn[:, j] = Slerp(t_src, Rotation.from_matrix(G[:, j]))(t_new).as_matrix()
    return JWn, Gn


def build(npz, start=None, end=None, loops=1, fps=30, speed=1.0):
    d = np.load(npz)
    JW, G, rest = d["JW"].astype(np.float64), d["G"].astype(np.float64), d["rest"]
    src_fps = float(d["fps"])
    assert abs(src_fps - fps) < 1e-3, f"bake at {src_fps} fps, want {fps}"
    a = int(round((start or 0) * fps))
    b = int(round(end * fps)) if end else len(JW)
    JW, G = JW[a:b], G[a:b]
    if speed != 1.0:
        JW, G = retime(JW, G, speed)
    if loops > 1:  # tile, carrying the horizontal drift of each pass
        drift = JW[-1, 0] - JW[0, 0]
        drift[1] = 0
        JW = np.concatenate([JW + k * drift for k in range(loops)])
        G = np.concatenate([G] * loops)
    ank = 0.5 * (rest[7, 1] + rest[8, 1])
    s = SOMA_HIP_TO_ANKLE / (rest[0, 1] - ank)
    idx = [SMPL_OF[n] for n in MJCF_BODY_NAMES]
    Gz = np.einsum("ij,ftjk,lk->ftil", M, G[:, idx], M)  # M G M^T
    root = (JW[:, 0] - np.array([JW[0, 0, 0], 0, JW[0, 0, 2]])) * s
    root = root @ M.T

    kin = extract_kinematic_info(MJCF)
    assert list(kin.body_names) == MJCF_BODY_NAMES, kin.body_names
    g = torch.tensor(Gz, dtype=torch.float32)
    r = torch.tensor(root, dtype=torch.float32)
    local = compute_joint_rot_mats_from_global_mats(kinematic_info=kin, global_rot_mats=g)

    def make(rootpos):
        m = fk_from_transforms_with_velocities(
            kinematic_info=kin, root_pos=rootpos, joint_rot_mats=local, fps=fps,
            compute_velocities=True, velocity_max_horizon=3)
        return m

    m = make(r)
    # floor: lift/drop so the lowest contact sits on z=0 (2nd percentile)
    pos = m.rigid_body_pos.numpy()
    gaps = np.stack([pos[:, MJCF_BODY_NAMES.index(n), 2] - c for n, c in CLEAR.items()], 1)
    dz = float(np.percentile(gaps.min(1), 2.0))
    r[:, 2] -= dz
    m = make(r)
    m.local_rigid_body_rot = matrix_to_quaternion(local, w_last=True)
    qpos = extract_qpos_from_transforms(kinematic_info=kin, root_pos=r,
                                        joint_rot_mats=local,
                                        multi_dof_decomposition_method="exp_map")
    m.dof_pos = qpos[:, 7:]
    nj = len(MJCF_BODY_NAMES) - 1
    m.dof_vel = compute_angular_velocity(batched_robot_rot_mats=local[:, 1:], fps=fps).reshape(-1, nj * 3)
    m.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=m.rigid_body_pos, velocity=m.rigid_body_vel,
        vel_thres=0.15, height_thresh=0.1).to(torch.bool)
    info = {"frames": int(len(r)), "scale": float(s), "floor_shift": dz,
            "min_body_z": float(m.rigid_body_pos[..., 2].min())}
    return m, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    with torch.no_grad():
        m, info = build(a.npz, a.start, a.end, a.loops, speed=a.speed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.to_dict(), a.out)
    print("[soma_from_bake]", info)
