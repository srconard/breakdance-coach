"""Motion cleanup for GVHMR SMPL sequences — Phase 1 of the 3D quality plan.

Cleans raw GVHMR mocap before Blender export. Video mocap jitters, and
breaking is its adversarial case: inverted poses, hands-as-feet, motion blur.
This module makes the skeleton animation smooth, grounded, and believable:

    raw .pkl
      -> temporal smoothing        (quaternions per joint + root translation,
                                    Savitzky-Golay in the right space)
      -> contact detection         (feet AND hands: low speed + near floor)
      -> contact locking           (root anti-slide correction, then per-limb
                                    analytic two-bone IK pinning)
      -> floor clamping            (per-frame min-vertex offset, floor -> y=0)
      -> cleaned .pkl              (same format; Blender script needs no changes)

All math is pure numpy on the SMPL global params (Y-up world space, floor is
a y=const plane). `smpl_params_incam` is passed through untouched (it becomes
stale relative to the cleaned global params).

Usage:
    python -m analyzer_3d.src.motion_cleanup input_smpl.pkl
    python -m analyzer_3d.src.motion_cleanup input_smpl.pkl -o cleaned.pkl --report
    python -m analyzer_3d.src.motion_cleanup input_smpl.pkl --no-ik --no-floor

Or from code:
    from analyzer_3d.src.motion_cleanup import clean_pkl
    clean_pkl("raw_smpl.pkl", "clean_smpl.pkl")
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# SMPL kinematic tree for all 24 joints (hands 22/23 hang off the wrists)
PARENTS24 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# End effectors we track for contact: name -> (joint, child_tip, ik_chain)
# In breaking, hands are feet — wrists get the same treatment as ankles.
# ik_chain = (root, mid, effector) for the analytic two-bone solve.
EFFECTORS = {
    "L_ankle": {"joint": 7, "tip": 10, "chain": (1, 4, 7)},
    "R_ankle": {"joint": 8, "tip": 11, "chain": (2, 5, 8)},
    "L_wrist": {"joint": 20, "tip": 22, "chain": (16, 18, 20)},
    "R_wrist": {"joint": 21, "tip": 23, "chain": (17, 19, 21)},
}

# Contact thresholds (meters above floor, using min(joint, tip) height)
HEIGHT_THRESH = {"L_ankle": 0.10, "R_ankle": 0.10, "L_wrist": 0.09, "R_wrist": 0.09}
SPEED_THRESH = 0.45          # m/s — effector must be near-stationary to count as planted
MIN_CONTACT_SEC = 0.10       # segments shorter than this are noise
IK_RAMP_SEC = 0.10           # blend IK in/out over this many seconds at segment edges


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities (numpy, vectorized, wxyz convention)
# ---------------------------------------------------------------------------

def aa_to_quat(aa):
    """Axis-angle (..., 3) -> quaternion (..., 4) wxyz."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = angle * 0.5
    # sin(x)/x is stable via sinc: np.sinc(x/pi) = sin(x)/x
    k = 0.5 * np.sinc(half / np.pi)
    q = np.concatenate([np.cos(half), aa * k], axis=-1)
    return q


def quat_to_aa(q):
    """Quaternion (..., 4) wxyz -> axis-angle (..., 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w = np.clip(q[..., :1], -1.0, 1.0)
    angle = 2.0 * np.arccos(np.abs(w))
    sign = np.where(q[..., :1] < 0, -1.0, 1.0)  # keep angle in [0, pi]
    xyz = q[..., 1:] * sign
    s = np.linalg.norm(xyz, axis=-1, keepdims=True)
    axis = np.where(s > 1e-8, xyz / np.maximum(s, 1e-12), np.array([1.0, 0.0, 0.0]))
    return axis * angle


def quat_hemisphere_align(q):
    """Flip signs along time axis 0 so consecutive quats are in the same hemisphere."""
    q = q.copy()
    for f in range(1, q.shape[0]):
        dots = np.sum(q[f] * q[f - 1], axis=-1, keepdims=True)
        q[f] = np.where(dots < 0, -q[f], q[f])
    return q


def rodrigues(aa):
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3)."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = aa / np.maximum(angle, 1e-12)
    a = angle[..., None]
    cos, sin = np.cos(a), np.sin(a)
    x, y, z = axis[..., 0, None, None], axis[..., 1, None, None], axis[..., 2, None, None]
    zero = np.zeros_like(x)
    K = np.concatenate([
        np.concatenate([zero, -z, y], axis=-1),
        np.concatenate([z, zero, -x], axis=-1),
        np.concatenate([-y, x, zero], axis=-1),
    ], axis=-2)
    eye = np.broadcast_to(np.eye(3), K.shape)
    R = eye * cos + sin * K + (1 - cos) * (axis[..., :, None] * axis[..., None, :])
    # Identity where angle ~ 0
    small = (angle[..., None] < 1e-8)
    return np.where(small, eye, R)


def mat_to_quat(R):
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4) wxyz (Shepperd's method)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    tr = m00 + m11 + m22

    q = np.empty(R.shape[:-2] + (4,), dtype=R.dtype)
    # branch masks
    c0 = tr > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)

    s = np.sqrt(np.where(c0, tr + 1.0, 1.0)) * 2.0
    q0 = np.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], axis=-1)
    s = np.sqrt(np.where(c1, 1.0 + m00 - m11 - m22, 1.0)) * 2.0
    q1 = np.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], axis=-1)
    s = np.sqrt(np.where(c2, 1.0 + m11 - m00 - m22, 1.0)) * 2.0
    q2 = np.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], axis=-1)
    s = np.sqrt(np.where(c3, 1.0 + m22 - m00 - m11, 1.0)) * 2.0
    q3 = np.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], axis=-1)

    q = np.where(c0[..., None], q0, np.where(c1[..., None], q1, np.where(c2[..., None], q2, q3)))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def mat_to_aa(R):
    return quat_to_aa(mat_to_quat(R))


def quat_slerp(q0, q1, t):
    """Slerp between quat arrays (broadcasting); t in [0,1] (..., 1) or scalar."""
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / np.maximum(sin_theta, 1e-12))
    w1 = np.where(near, t, np.sin(t * theta) / np.maximum(sin_theta, 1e-12))
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def rot_between(v0, v1):
    """Rotation matrix (3,3) taking direction v0 to v1 (both 3-vectors)."""
    a = v0 / max(np.linalg.norm(v0), 1e-12)
    b = v1 / max(np.linalg.norm(v1), 1e-12)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a
        perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return rodrigues(perp * np.pi)
    axis = axis / s
    angle = np.arctan2(s, c)
    return rodrigues(axis * angle)


# ---------------------------------------------------------------------------
# Savitzky-Golay filter (no scipy dependency)
# ---------------------------------------------------------------------------

def _savgol_coeffs(window, polyorder):
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, polyorder + 1, increasing=True)  # (window, poly+1)
    # coefficients that evaluate the fitted polynomial at x=0
    pinv = np.linalg.pinv(A)
    return pinv[0]  # (window,)


def savgol(x, window, polyorder=3, axis=0):
    """Savitzky-Golay smoothing along `axis` with mirror-padded edges."""
    window = int(window)
    if window % 2 == 0:
        window += 1
    n = x.shape[axis]
    if window < polyorder + 2:
        window = polyorder + 2 + ((polyorder + 2) % 2 == 0)  # next odd
    if window >= n:
        window = n if n % 2 == 1 else n - 1
    if window < 3:
        return x.copy()
    poly = min(polyorder, window - 1)
    coeffs = _savgol_coeffs(window, poly)

    x_moved = np.moveaxis(x, axis, 0)
    half = window // 2
    padded = np.concatenate([x_moved[1:half + 1][::-1], x_moved, x_moved[-half - 1:-1][::-1]], axis=0)
    flat = padded.reshape(padded.shape[0], -1)
    out = np.empty((n, flat.shape[1]), dtype=np.float64)
    for col in range(flat.shape[1]):
        out[:, col] = np.convolve(flat[:, col], coeffs[::-1], mode="valid")
    out = out.reshape((n,) + x_moved.shape[1:])
    return np.moveaxis(out, 0, axis).astype(x.dtype)


def median_filter_bool(mask, width=3):
    """Majority-vote filter for boolean 1D arrays (removes 1-frame flickers)."""
    if width < 3:
        return mask
    half = width // 2
    padded = np.pad(mask.astype(np.int32), half, mode="edge")
    out = np.zeros_like(mask)
    for i in range(len(mask)):
        out[i] = padded[i:i + width].sum() > half
    return out.astype(bool)


# ---------------------------------------------------------------------------
# SMPL forward kinematics + LBS
# ---------------------------------------------------------------------------

def full_pose_aa(body_pose, global_orient):
    """Stack into (F, 24, 3) axis-angle; hands (22, 23) get identity."""
    F = body_pose.shape[0]
    return np.concatenate(
        [global_orient[:, None, :], body_pose.reshape(F, 21, 3), np.zeros((F, 2, 3), dtype=body_pose.dtype)],
        axis=1,
    )


def fk(body_pose, global_orient, transl, rest_joints):
    """SMPL forward kinematics.

    Returns:
        joints_world: (F, 24, 3) posed joint positions (transl applied)
        G: (F, 24, 3, 3) global (world) rotation of each joint
        pos: (F, 24, 3) posed joint positions before transl
    """
    aa = full_pose_aa(body_pose, global_orient)
    R = rodrigues(aa)  # (F, 24, 3, 3)
    F = R.shape[0]
    G = np.zeros_like(R)
    pos = np.zeros((F, 24, 3), dtype=np.float64)
    G[:, 0] = R[:, 0]
    pos[:, 0] = rest_joints[0]
    for i in range(1, 24):
        p = PARENTS24[i]
        G[:, i] = G[:, p] @ R[:, i]
        offset = (rest_joints[i] - rest_joints[p]).astype(np.float64)
        pos[:, i] = pos[:, p] + np.einsum("fab,b->fa", G[:, p], offset)
    joints_world = pos + transl[:, None, :]
    return joints_world, G, pos


def lbs_min_height(G, pos, transl, mesh, rest_joints, stride=3):
    """Per-frame minimum vertex height (world y) via linear blend skinning.

    Vertices are strided for speed — plenty for a floor-penetration estimate.
    """
    verts = mesh["vertices"][::stride].astype(np.float64)   # (V, 3)
    W = mesh["weights"][::stride].astype(np.float64)         # (V, 24)
    F = G.shape[0]
    min_y = np.empty(F)
    for f in range(F):
        Rj = G[f]                                # (24, 3, 3)
        tj = pos[f] - np.einsum("jab,jb->ja", Rj, rest_joints)  # (24, 3)
        # y-row only: v_y = sum_j w_vj * (R_j[1] . v + t_j[1])
        ry = np.einsum("ja,va->jv", Rj[:, 1, :], verts) + tj[:, 1:2]  # (24, V)
        vy = np.einsum("vj,jv->v", W, ry)
        min_y[f] = vy.min() + transl[f, 1]
    return min_y


# ---------------------------------------------------------------------------
# Cleanup stages
# ---------------------------------------------------------------------------

def smooth_rotations(body_pose, global_orient, fps, window_sec=0.20):
    """Savitzky-Golay smoothing of per-joint rotations in quaternion space."""
    F = body_pose.shape[0]
    aa = full_pose_aa(body_pose, global_orient)[:, :22]  # (F, 22, 3)
    q = aa_to_quat(aa)                                   # (F, 22, 4)
    q = quat_hemisphere_align(q)
    window = max(3, int(round(window_sec * fps)) | 1)
    q_s = savgol(q, window, polyorder=3, axis=0)
    q_s = q_s / np.linalg.norm(q_s, axis=-1, keepdims=True)
    aa_s = quat_to_aa(q_s)
    return aa_s[:, 1:22].reshape(F, 63), aa_s[:, 0]


def detect_contacts(joints_world, floor_y, fps):
    """Detect contact segments for feet and hands.

    Returns dict effector_name -> list of (start_frame, end_frame_exclusive).
    """
    F = joints_world.shape[0]
    dt = 1.0 / fps
    min_len = max(2, int(round(MIN_CONTACT_SEC * fps)))
    segments = {}
    for name, cfg in EFFECTORS.items():
        j, tip = cfg["joint"], cfg["tip"]
        height = np.minimum(joints_world[:, j, 1], joints_world[:, tip, 1]) - floor_y
        vel = np.zeros(F)
        vel[1:] = np.linalg.norm(np.diff(joints_world[:, j], axis=0), axis=-1) / dt
        vel[0] = vel[1] if F > 1 else 0.0
        mask = (height < HEIGHT_THRESH[name]) & (vel < SPEED_THRESH)
        mask = median_filter_bool(mask, 3)
        # extract segments
        segs = []
        start = None
        for f in range(F):
            if mask[f] and start is None:
                start = f
            elif not mask[f] and start is not None:
                if f - start >= min_len:
                    segs.append((start, f))
                start = None
        if start is not None and F - start >= min_len:
            segs.append((start, F))
        segments[name] = segs
    return segments


def root_antislide(transl, joints_world, segments, fps):
    """Horizontal (XZ) root correction so planted effectors stop drifting.

    For every frame with active contacts, computes the mean XZ drift of the
    contacting effectors relative to their positions at contact start, and
    shifts the root to cancel it. Held (not decayed) while airborne, then
    smoothed so the trajectory stays fluid — exact pinning is the IK's job.
    """
    F = transl.shape[0]
    c = np.zeros((F, 2))  # XZ correction
    anchors = {}
    seg_start = {name: {s: (s, e) for (s, e) in segs} for name, segs in segments.items()}
    active_prev = set()
    for f in range(F):
        drifts = []
        for name, segs in segments.items():
            for (s, e) in segs:
                if s <= f < e:
                    key = (name, s)
                    if key not in anchors:
                        base = joints_world[s, EFFECTORS[name]["joint"]][[0, 2]]
                        anchors[key] = base + (c[f - 1] if f > 0 else 0.0)
                    pos_xz = joints_world[f, EFFECTORS[name]["joint"]][[0, 2]]
                    drifts.append(anchors[key] - pos_xz)
        if drifts:
            c[f] = np.mean(drifts, axis=0)
        elif f > 0:
            c[f] = c[f - 1]
    window = max(3, int(round(0.25 * fps)) | 1)
    c = savgol(c, window, polyorder=2, axis=0)
    out = transl.copy()
    out[:, 0] += c[:, 0]
    out[:, 2] += c[:, 1]
    return out, c


def two_bone_ik_frame(a, b, c, target, G, Ra_local, Rb_local, ja, jb):
    """Analytic two-bone IK for one frame.

    a, b, c: current world positions of chain root / mid / effector.
    target:  desired world position for c.
    G:       (24, 3, 3) global rotations this frame (pre-IK).
    Ra_local, Rb_local: current local rotation matrices of joints ja, jb.
    Returns new local rotation matrices (Ra', Rb').
    """
    l1 = np.linalg.norm(b - a)
    l2 = np.linalg.norm(c - b)
    d = np.linalg.norm(target - a)
    d = np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4)

    # bend axis from current pose (preserves the knee/elbow plane)
    n = np.cross(b - a, c - a)
    if np.linalg.norm(n) < 1e-8:
        n = np.cross(b - a, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(n) < 1e-8:
            n = np.cross(b - a, np.array([1.0, 0.0, 0.0]))
    n = n / np.linalg.norm(n)

    # interior angle at the mid joint (law of cosines)
    cos_cur = np.clip((l1**2 + l2**2 - np.linalg.norm(c - a)**2) / (2 * l1 * l2), -1.0, 1.0)
    cos_new = np.clip((l1**2 + l2**2 - d**2) / (2 * l1 * l2), -1.0, 1.0)
    delta = np.arccos(cos_new) - np.arccos(cos_cur)

    # try both rotation signs, keep whichever hits the desired reach
    best = None
    for sign in (1.0, -1.0):
        dR = rodrigues(n * (sign * delta))
        c_try = b + dR @ (c - b)
        err = abs(np.linalg.norm(c_try - a) - d)
        if best is None or err < best[0]:
            best = (err, dR, c_try)
    _, dR1, c_new = best

    # swing the whole chain so the effector lands on the target
    dR2 = rot_between(c_new - a, target - a)

    Gp_b = G[PARENTS24[jb]]  # global rotation of ja (mid's parent) pre-IK
    Gp_a = G[PARENTS24[ja]]
    Rb_new = Gp_b.T @ dR1 @ Gp_b @ Rb_local
    Ra_new = Gp_a.T @ dR2 @ Gp_a @ Ra_local
    return Ra_new, Rb_new


def apply_contact_ik(body_pose, global_orient, transl, rest_joints, segments, fps):
    """Pin contacting effectors to per-segment anchors with blended two-bone IK."""
    F = body_pose.shape[0]
    joints_world, G, _ = fk(body_pose, global_orient, transl, rest_joints)
    aa_full = full_pose_aa(body_pose, global_orient)
    R_local = rodrigues(aa_full)  # (F, 24, 3, 3)
    q_orig = aa_to_quat(aa_full)

    new_bp = body_pose.copy().astype(np.float64)
    ramp = max(2, int(round(IK_RAMP_SEC * fps)))

    for name, segs in segments.items():
        ja, jb, je = EFFECTORS[name]["chain"]
        for (s, e) in segs:
            seg = joints_world[s:e, je]
            anchor = np.median(seg, axis=0)
            for f in range(s, e):
                # blend weight: ramp in/out at segment edges
                w = min(1.0, (f - s + 1) / ramp, (e - f) / ramp)
                if w <= 0:
                    continue
                a = joints_world[f, ja]
                b = joints_world[f, jb]
                cpos = joints_world[f, je]
                Ra_new, Rb_new = two_bone_ik_frame(
                    a, b, cpos, anchor, G[f],
                    R_local[f, ja], R_local[f, jb], ja, jb,
                )
                qa = quat_slerp(q_orig[f, ja], mat_to_quat(Ra_new), w)
                qb = quat_slerp(q_orig[f, jb], mat_to_quat(Rb_new), w)
                new_bp[f, (ja - 1) * 3:(ja - 1) * 3 + 3] = quat_to_aa(qa)
                new_bp[f, (jb - 1) * 3:(jb - 1) * 3 + 3] = quat_to_aa(qb)
    return new_bp.astype(body_pose.dtype)


def clamp_floor(body_pose, global_orient, transl, rest_joints, mesh, floor_y, fps):
    """Raise the root so no vertex dips below the floor; smooth the offset."""
    _, G, pos = fk(body_pose, global_orient, transl, rest_joints)
    min_y = lbs_min_height(G, pos, transl, mesh, rest_joints)
    required = np.clip(floor_y - min_y, 0.0, None)
    window = max(5, int(round(0.45 * fps)) | 1)
    offset = required.copy()
    for _ in range(4):
        offset = np.maximum(savgol(offset, window, polyorder=2, axis=0), required)
    out = transl.copy()
    out[:, 1] += offset
    return out, required, min_y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def motion_metrics(body_pose, global_orient, transl, rest_joints, fps, floor_y, mesh, segments=None):
    """Quality metrics for a sequence. Returns a dict."""
    dt = 1.0 / fps
    joints_world, G, pos = fk(body_pose, global_orient, transl, rest_joints)

    q = aa_to_quat(full_pose_aa(body_pose, global_orient)[:, :22])
    q = quat_hemisphere_align(q)
    dots = np.abs(np.clip(np.sum(q[1:] * q[:-1], axis=-1), -1.0, 1.0))
    ang_vel = 2.0 * np.arccos(dots) / dt                       # (F-1, 22) rad/s
    ang_acc = np.abs(np.diff(ang_vel, axis=0)) / dt            # (F-2, 22) rad/s^2

    trans_acc = np.linalg.norm(np.diff(transl, n=2, axis=0), axis=-1) / dt**2

    min_y = lbs_min_height(G, pos, transl, mesh, rest_joints, stride=6)
    penetration = np.clip(floor_y - min_y, 0.0, None)

    slide = []
    if segments:
        for name, segs in segments.items():
            j = EFFECTORS[name]["joint"]
            for (s, e) in segs:
                if e - s >= 2:
                    xz = joints_world[s:e, j][:, [0, 2]]
                    slide.append(np.linalg.norm(np.diff(xz, axis=0), axis=-1).mean() / dt)

    return {
        "rot_jitter_deg_s2": float(np.degrees(ang_acc.mean())),
        "transl_jitter_m_s2": float(trans_acc.mean()),
        "penetration_mean_cm": float(penetration.mean() * 100),
        "penetration_max_cm": float(penetration.max() * 100),
        "contact_slide_cm_s": float(np.mean(slide) * 100) if slide else 0.0,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def clean_pkl(
    input_pkl,
    output_pkl=None,
    smooth_window_sec=0.20,
    do_ik=True,
    do_floor=True,
    rebase_floor=True,
    report=True,
):
    """Run the full Phase 1 cleanup on a GVHMR pickle. Returns output path."""
    input_pkl = Path(input_pkl)
    if output_pkl is None:
        output_pkl = input_pkl.with_name(input_pkl.stem + "_clean.pkl")
    output_pkl = Path(output_pkl)

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    params = data["smpl_params_global"]
    fps = float(data.get("fps", 30.0))
    if "mesh" not in data:
        raise ValueError("Pickle has no 'mesh' block (rest joints needed for cleanup)")
    rest_joints = data["mesh"]["joints"].astype(np.float64)
    mesh = data["mesh"]

    bp = params["body_pose"].astype(np.float64)
    go = params["global_orient"].astype(np.float64)
    tr = params["transl"].astype(np.float64)
    F = bp.shape[0]
    print(f"Motion cleanup: {input_pkl.name} ({F} frames @ {fps:g} fps)")

    # Floor estimate from the raw sequence (robust: median of per-frame min height)
    _, G0, pos0 = fk(bp, go, tr, rest_joints)
    min_y_raw = lbs_min_height(G0, pos0, tr, mesh, rest_joints)
    floor_y = float(np.median(min_y_raw))
    print(f"  Estimated floor height: y = {floor_y:.3f} m")

    before = motion_metrics(bp, go, tr, rest_joints, fps, floor_y, mesh)

    # 1. Temporal smoothing (quaternion space + root translation)
    print(f"  [1/4] Smoothing rotations (SG window {smooth_window_sec:.2f}s) + translation...")
    bp, go = smooth_rotations(bp, go, fps, smooth_window_sec)
    tr_window = max(3, int(round(smooth_window_sec * fps)) | 1)
    tr = savgol(tr, tr_window, polyorder=3, axis=0)

    # 2. Contact detection on the smoothed motion
    joints_world, _, _ = fk(bp, go, tr, rest_joints)
    segments = detect_contacts(joints_world, floor_y, fps)
    n_segs = sum(len(s) for s in segments.values())
    print(f"  [2/4] Contacts: " + ", ".join(f"{k}:{len(v)}" for k, v in segments.items())
          + f" ({n_segs} segments)")

    # 3. Contact locking: root anti-slide, then per-limb IK pinning
    if n_segs:
        tr, _ = root_antislide(tr, joints_world, segments, fps)
        if do_ik:
            print(f"  [3/4] Two-bone IK pinning on {n_segs} contact segments...")
            joints_world, _, _ = fk(bp, go, tr, rest_joints)
            segments = detect_contacts(joints_world, floor_y, fps)
            bp = apply_contact_ik(bp, go, tr, rest_joints, segments, fps)
    else:
        print("  [3/4] No contacts detected — skipping locking")

    # 4. Floor clamping
    if do_floor:
        print("  [4/4] Floor clamping...")
        tr, required, _ = clamp_floor(bp, go, tr, rest_joints, mesh, floor_y, fps)
        if required.max() > 0:
            print(f"      max penetration fixed: {required.max()*100:.1f} cm")

    # Re-base so the floor sits at y=0 (viewers and future ground planes assume this)
    if rebase_floor:
        tr[:, 1] -= floor_y
        floor_out = 0.0
    else:
        floor_out = floor_y

    after = motion_metrics(bp, go, tr, rest_joints, fps, floor_out, mesh, segments)
    # 'before' metrics had no segments; recompute slide on raw data with same segments
    before_slide = motion_metrics(
        params["body_pose"].astype(np.float64),
        params["global_orient"].astype(np.float64),
        params["transl"].astype(np.float64),
        rest_joints, fps, floor_y, mesh, segments,
    )
    before["contact_slide_cm_s"] = before_slide["contact_slide_cm_s"]

    if report:
        print("\n  Metric                    before      after")
        rows = [
            ("rot jitter (deg/s^2)", "rot_jitter_deg_s2"),
            ("transl jitter (m/s^2)", "transl_jitter_m_s2"),
            ("penetration mean (cm)", "penetration_mean_cm"),
            ("penetration max (cm)", "penetration_max_cm"),
            ("contact slide (cm/s)", "contact_slide_cm_s"),
        ]
        for label, key in rows:
            print(f"  {label:<25} {before[key]:>9.2f}  {after[key]:>9.2f}")
        print()

    out = dict(data)
    out["smpl_params_global"] = {
        "body_pose": bp.astype(np.float32),
        "global_orient": go.astype(np.float32),
        "transl": tr.astype(np.float32),
        "betas": params["betas"],
    }
    out["cleanup"] = {
        "floor_y_original": floor_y,
        "rebased_to_zero": bool(rebase_floor),
        "segments": {k: v for k, v in segments.items()},
        "metrics_before": before,
        "metrics_after": after,
    }
    with open(output_pkl, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {output_pkl}")
    return output_pkl


def main():
    parser = argparse.ArgumentParser(description="Clean raw GVHMR mocap (Phase 1)")
    parser.add_argument("input", help="Input SMPL pickle from GVHMR")
    parser.add_argument("-o", "--output", default=None, help="Output pickle (default: <stem>_clean.pkl)")
    parser.add_argument("--smooth-window", type=float, default=0.20, help="SG window in seconds")
    parser.add_argument("--no-ik", action="store_true", help="Skip two-bone IK contact pinning")
    parser.add_argument("--no-floor", action="store_true", help="Skip floor clamping")
    parser.add_argument("--no-rebase", action="store_true", help="Keep original floor height (don't move floor to y=0)")
    args = parser.parse_args()

    clean_pkl(
        args.input,
        args.output,
        smooth_window_sec=args.smooth_window,
        do_ik=not args.no_ik,
        do_floor=not args.no_floor,
        rebase_floor=not args.no_rebase,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
