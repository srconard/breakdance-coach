"""Motion-matching feature extraction + transition search (choreo v1 seams).

Implements the classic motion-matching feature vector (Clavet GDC 2016 /
orangeduck reference impl) over SMPL joint positions, and a Motion-Graphs
style windowed transition search (Kovar 2002 point-cloud-window distance,
computed on feature vectors instead of skinned point clouds): given the
tail of clip A and the head of clip B, find the frame pair (i, j) where
cutting A after i and starting B at j is least visible.

Per-frame features, all in the character's root-local yaw-normalized frame
(invariant to the rigid yaw+XZ retarget choreo.py applies later):
    - end-effector positions   feet/wrists/head rel. to ground-projected root
    - end-effector velocities  same joints, world velocity in local frame
    - root height              pelvis y (floor is y=0 in cleaned clips)
    - root linear velocity     pelvis velocity in local frame
    - root angular velocity    yaw rate
35 dims total, z-score normalized per dimension over the working library
(the set of clips passed to feature_stats).

Distance between candidate cut frames is the mean squared feature distance
over a +/-window around the pair (windowed distance stabilizes the match:
a single-frame minimum can sit inside a velocity mismatch); reported values
are the sqrt (RMS distance in normalized feature units).

Usage (report best transitions between consecutive clips):
    python -m analyzer_3d.src.motion_match A_smpl_clean.pkl B_smpl_clean.pkl
        [more.pkl ...] [--exit-s 1.5] [--entry-s 1.5] [--window-s 0.15]
        [--top 3]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from .motion_cleanup import fk

UP = np.array([0.0, 1.0, 0.0])

# SMPL joints used as end effectors: L/R ankle, L/R wrist, head.
# In breaking, hands are feet — wrists matter as much as ankles.
EE_JOINTS = [7, 8, 20, 21, 15]
EE_NAMES = ["L_ankle", "R_ankle", "L_wrist", "R_wrist", "head"]

FEATURE_DIM = len(EE_JOINTS) * 6 + 5  # pos3 + vel3 per EE, +root h/vel3/yawrate


def yaw_of(jw):
    """Per-frame yaw of the hips, radians; 0 = facing +Z, CCW about +Y."""
    lat = jw[:, 1, :] - jw[:, 2, :]          # L_hip - R_hip
    fwd = np.cross(lat, UP[None, :])
    return np.arctan2(fwd[:, 0], fwd[:, 2])


def extract_features(jw, fps):
    """Motion-matching feature vectors for a clip.

    jw: (T, 24, 3) world joint positions from fk(). Returns (T, 35) float64,
    unnormalized (see feature_stats / z-score before comparing clips).
    """
    T = jw.shape[0]
    root = jw[:, 0, :]                                   # pelvis
    yaw = yaw_of(jw)
    c, s = np.cos(yaw), np.sin(yaw)

    def to_local(v):
        """Rotate world vectors (T, ..., 3) by R_y(-yaw) into character frame."""
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        shape = (T,) + (1,) * (v.ndim - 2)
        cc, ss = c.reshape(shape), s.reshape(shape)
        return np.stack([cc * x - ss * z, y, ss * x + cc * z], axis=-1)

    ground_root = root * np.array([1.0, 0.0, 1.0])       # project to floor
    ee_pos = to_local(jw[:, EE_JOINTS, :] - ground_root[:, None, :])
    ee_vel = to_local(np.gradient(jw[:, EE_JOINTS, :], axis=0) * fps)
    root_vel = to_local(np.gradient(root, axis=0) * fps)
    yaw_rate = np.gradient(np.unwrap(yaw)) * fps

    return np.concatenate([
        ee_pos.reshape(T, -1),
        ee_vel.reshape(T, -1),
        root[:, 1:2],
        root_vel,
        yaw_rate[:, None],
    ], axis=1)


def feature_stats(feats):
    """Per-dimension z-score stats over the library. feats: (N, 35) stacked."""
    mean = feats.mean(axis=0)
    std = np.maximum(feats.std(axis=0), 1e-6)
    return mean, std


def windowed_distance(fA, fB, exit_lo, exit_hi, entry_lo, entry_hi, win):
    """RMS feature distance for every candidate cut pair.

    fA, fB: normalized features (T, D). Candidate exit frames of A are
    [exit_lo, exit_hi), entry frames of B are [entry_lo, entry_hi). The
    distance at (i, j) averages squared distances over offsets w in
    [-win, +win] where both i+w and j+w are valid frames (Kovar-style
    window, diagonal so velocities/phase must agree, not just the pose).
    Returns (ni, nj) matrix of RMS distances.
    """
    TA, TB = len(fA), len(fB)
    ia0, ia1 = max(0, exit_lo - win), min(TA, exit_hi + win)
    jb0, jb1 = max(0, entry_lo - win), min(TB, entry_hi + win)
    a, b = fA[ia0:ia1], fB[jb0:jb1]
    D0 = ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2.0 * (a @ b.T))
    np.maximum(D0, 0.0, out=D0)

    I = np.arange(exit_lo, exit_hi)[:, None]
    J = np.arange(entry_lo, entry_hi)[None, :]
    acc = np.zeros((exit_hi - exit_lo, entry_hi - entry_lo))
    cnt = np.zeros_like(acc)
    for w in range(-win, win + 1):
        valid = (I + w >= 0) & (I + w < TA) & (J + w >= 0) & (J + w < TB)
        iw = np.clip(I + w - ia0, 0, D0.shape[0] - 1)
        jw = np.clip(J + w - jb0, 0, D0.shape[1] - 1)
        acc += np.where(valid, D0[iw, jw], 0.0)
        cnt += valid
    return np.sqrt(acc / np.maximum(cnt, 1) / fA.shape[1])


def find_transition(fA, fB, exit_lo, exit_hi, entry_lo, entry_hi, win):
    """Best cut pair between clip A's tail and clip B's head.

    Returns (i, j, dist, D): cut A after frame i, start B at frame j;
    dist is the windowed RMS feature distance; D is the full candidate
    matrix (indexed [i - exit_lo, j - entry_lo]) for reporting.
    """
    D = windowed_distance(fA, fB, exit_lo, exit_hi, entry_lo, entry_hi, win)
    di, dj = np.unravel_index(np.argmin(D), D.shape)
    return exit_lo + int(di), entry_lo + int(dj), float(D[di, dj]), D


def load_features(pkl_path):
    """Load a library pkl -> (features (T, 35), fps)."""
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    g = pkl["smpl_params_global"]
    fps = float(pkl["fps"])
    rest = pkl["mesh"]["joints"].astype(np.float64)
    jw, _, _ = fk(g["body_pose"].astype(np.float64),
                  g["global_orient"].astype(np.float64),
                  g["transl"].astype(np.float64), rest)
    return extract_features(jw, fps), fps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clips", nargs="+", help="two or more *_smpl_clean.pkl")
    ap.add_argument("--exit-s", type=float, default=1.5,
                    help="search window at the end of the earlier clip")
    ap.add_argument("--entry-s", type=float, default=1.5,
                    help="search window at the start of the later clip")
    ap.add_argument("--window-s", type=float, default=0.15,
                    help="+/- seconds for the windowed distance")
    ap.add_argument("--top", type=int, default=3, help="matches to print per pair")
    args = ap.parse_args()
    if len(args.clips) < 2:
        ap.error("need at least two clips")

    feats, fps0 = [], None
    for p in args.clips:
        f, fps = load_features(p)
        if fps0 is None:
            fps0 = fps
        assert fps == fps0, f"fps mismatch: {p}"
        feats.append(f)
        print(f"[mm] {Path(p).stem}: {len(f)} frames @ {fps:g} fps")
    mean, std = feature_stats(np.concatenate(feats))
    feats = [(f - mean) / std for f in feats]

    win = int(round(args.window_s * fps0))
    for k in range(len(feats) - 1):
        fA, fB = feats[k], feats[k + 1]
        exit_lo = max(0, len(fA) - int(round(args.exit_s * fps0)))
        entry_hi = min(len(fB), int(round(args.entry_s * fps0)))
        i, j, d, D = find_transition(fA, fB, exit_lo, len(fA), 0, entry_hi, win)
        naive = float(D[-1, 0])   # fixed boundary: last frame of A -> first of B
        print(f"\n[mm] {Path(args.clips[k]).stem} -> {Path(args.clips[k + 1]).stem}")
        print(f"[mm]   fixed boundary (end->start): dist {naive:.3f}")
        flat = np.argsort(D, axis=None)[:args.top]
        for rank, idx in enumerate(flat):
            di, dj = np.unravel_index(idx, D.shape)
            ii, jj = exit_lo + int(di), int(dj)
            print(f"[mm]   #{rank + 1}: cut A @ {ii / fps0:6.2f}s -> B @ "
                  f"{jj / fps0:5.2f}s  dist {float(D[di, dj]):.3f}"
                  f"  ({(1 - float(D[di, dj]) / max(naive, 1e-9)) * 100:+.0f}% vs fixed)")


if __name__ == "__main__":
    sys.exit(main() or 0)
