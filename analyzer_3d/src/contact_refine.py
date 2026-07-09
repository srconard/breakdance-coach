"""Contact-clamped refinement of GVHMR world-space motion (Phase 1.5 v2).

Pipeline position:  GVHMR pkl (enriched: keypoints_2d + K_fullimg)
                    -> THIS -> motion_cleanup -> smpl_to_glb.

The core failure this fixes: GVHMR never commits weight-bearing hand
contacts on breaking footage — the support wrist floats 12-40 cm above the
floor. Post-hoc contact locking can't help because the reconstruction never
puts the hand near the floor. Here the contact interval is an *input*
(human/VLM-annotated gates refined by 2D keypoint evidence), and the
sequence is re-optimized so reprojection + floor contact + smoothness pull
the body into a pose that satisfies it.

Stages:
  1. Fit ONE rigid world->camera transform for the whole clip (static-cam
     premise) by Procrustes-aligning global-frame FK joints to incam-frame
     FK joints. GVHMR's per-frame implied camera wobbles — that wobble IS
     the world drift; a constant camera turns 2D reprojection into a
     world-frame corrector.
  2. Derive per-effector contact segments: coarse gate windows (JSON,
     same contract a VLM annotator produces) refined by 2D wrist/ankle
     keypoint evidence — planted = confident + stationary + low in image —
     at full 30 fps resolution.
  3. Optimize global_orient/transl (stage A), then + body_pose (stage B),
     with Adam against: confidence-weighted Geman-McClure reprojection,
     joint-acceleration smoothness, contact clamping (tip joint -> floor
     height, XZ-still), and stay-near-init anchors.

Usage:
    python -m analyzer_3d.src.contact_refine output/3d/phase1/flare_demo30_smpl_v2.pkl \
        --gates output/3d/phase1/flare_demo30_gates.json
    # optional: --contacts <file.json> to bypass derivation with hand-written
    # per-effector intervals: {"L_wrist": [[t0,t1],...], ...}
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .motion_cleanup import PARENTS24, fk as fk_np, lbs_min_height

# COCO-17 keypoint indices
COCO = {
    "L_shoulder": 5, "R_shoulder": 6, "L_elbow": 7, "R_elbow": 8,
    "L_wrist": 9, "R_wrist": 10, "L_hip": 11, "R_hip": 12,
    "L_knee": 13, "R_knee": 14, "L_ankle": 15, "R_ankle": 16,
}
# SMPL joint indices for the same body points
SMPL_FOR_COCO = {
    "L_shoulder": 16, "R_shoulder": 17, "L_elbow": 18, "R_elbow": 19,
    "L_wrist": 20, "R_wrist": 21, "L_hip": 1, "R_hip": 2,
    "L_knee": 4, "R_knee": 5, "L_ankle": 7, "R_ankle": 8,
}
REPROJ_PAIRS = [(SMPL_FOR_COCO[k], COCO[k]) for k in COCO]

# Leg joints in SMPL body_pose layout (joints 1..21 -> slot j-1). Used by
# --protect-legs: freeze these to raw GVHMR so reprojection (depth-ambiguous
# single-view) can't drag the 3D legs off the multi-view-consistent pose it
# already has. See 2026-07-05-leg-extension-investigation (Finding 2).
LEG_SMPL_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]  # hips, knees, ankles, feet
LEG_BODY_SLOTS = [j - 1 for j in LEG_SMPL_JOINTS]

# Effectors: contact is enforced on the distal "tip" joint (toes / hand)
EFFECTOR_TIPS = {"L_ankle": 10, "R_ankle": 11, "L_wrist": 22, "R_wrist": 23}
EFFECTOR_KP = {"L_ankle": 15, "R_ankle": 16, "L_wrist": 9, "R_wrist": 10}
TIP_FLOOR_OFFSET = {"L_ankle": 0.03, "R_ankle": 0.03, "L_wrist": 0.02, "R_wrist": 0.02}

# Contact derivation thresholds
KP_CONF_MIN = 0.35
KP_SPEED_MAX = 0.012     # fraction of bbox size per frame
MIN_CONTACT_SEC = 0.15
GAP_FILL_SEC = 0.10
RAMP_SEC = 0.10

# Loss weights (component magnitudes printed at iter 0 — tune there)
W_REPROJ = 1.0
W_CONTACT = 60.0
W_STILL = 150.0
W_SMOOTH = 400.0
W_ANCHOR_BODY = 20.0
W_ANCHOR_ORIENT = 2.0
W_ANCHOR_TRANSL = 0.3
GM_SIGMA = 0.04          # Geman-McClure scale, bbox-normalized units


# ---------------------------------------------------------------------------
# Torch SMPL FK
# ---------------------------------------------------------------------------

def rodrigues_t(aa):
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=aa.dtype).expand(*aa.shape[:-1], 3, 3)
    s = angle.sin()[..., None]
    c = angle.cos()[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def fk_t(body_pose, global_orient, transl, rest_joints):
    """Torch SMPL FK. Returns world joints (F, 24, 3)."""
    F = body_pose.shape[0]
    aa = torch.cat(
        [global_orient[:, None, :], body_pose.reshape(F, 21, 3),
         torch.zeros(F, 2, 3, dtype=body_pose.dtype)], dim=1)
    R = rodrigues_t(aa)                                   # (F, 24, 3, 3)
    G = [None] * 24
    pos = [None] * 24
    G[0] = R[:, 0]
    pos[0] = rest_joints[0].expand(F, 3)
    for i in range(1, 24):
        p = PARENTS24[i]
        G[i] = G[p] @ R[:, i]
        offset = rest_joints[i] - rest_joints[p]
        pos[i] = pos[p] + (G[p] @ offset[:, None]).squeeze(-1)
    joints = torch.stack(pos, dim=1) + transl[:, None, :]
    return joints


# ---------------------------------------------------------------------------
# Stage 1: constant world->camera transform
# ---------------------------------------------------------------------------

def fit_static_camera(pkl):
    """Procrustes-fit one rigid (R, t): world/global frame -> camera frame."""
    g, c = pkl["smpl_params_global"], pkl["smpl_params_incam"]
    rest = pkl["mesh"]["joints"].astype(np.float64)
    jg, _, _ = fk_np(g["body_pose"], g["global_orient"], g["transl"], rest)
    jc, _, _ = fk_np(c["body_pose"], c["global_orient"], c["transl"], rest)
    F = jg.shape[0]

    def solve(mask):
        X = jg[mask].reshape(-1, 3)
        Y = jc[mask].reshape(-1, 3)
        mx, my = X.mean(0), Y.mean(0)
        H = (X - mx).T @ (Y - my)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        t = my - R @ mx
        return R, t

    mask = np.ones(F, dtype=bool)
    R, t = solve(mask)
    # one trimming round: drop the worst 15% of frames (drift extremes)
    res = np.linalg.norm(jg @ R.T + t - jc, axis=2).mean(axis=1)
    mask = res <= np.quantile(res, 0.85)
    R, t = solve(mask)
    res2 = np.linalg.norm(jg @ R.T + t - jc, axis=2).mean(axis=1)
    print(f"[cam] static-camera fit: mean residual {res2.mean()*100:.1f} cm "
          f"(max {res2.max()*100:.1f} cm) — residual = GVHMR world drift")
    return R, t


# ---------------------------------------------------------------------------
# Stage 2: contact segments from gates + 2D evidence
# ---------------------------------------------------------------------------

def _intervals_to_mask(intervals, F, fps):
    m = np.zeros(F, dtype=bool)
    for a, b in intervals:
        m[int(round(a * fps)):int(round(b * fps)) + 1] = True
    return m


def _mask_to_intervals(mask, fps):
    out = []
    F = len(mask)
    i = 0
    while i < F:
        if mask[i]:
            j = i
            while j + 1 < F and mask[j + 1]:
                j += 1
            out.append([round(i / fps, 3), round(j / fps, 3)])
            i = j + 1
        else:
            i += 1
    return out


def derive_contacts(pkl, gates, fps):
    """Refine coarse gate windows into per-effector contact segments."""
    kp = pkl["keypoints_2d"]              # (F, 17, 3)
    bbx = pkl["bbx_xys"]                  # (F, 3) cx, cy, size
    F = kp.shape[0]
    min_len = max(2, int(round(MIN_CONTACT_SEC * fps)))
    gap = int(round(GAP_FILL_SEC * fps))

    gate_masks = {
        "wrist": _intervals_to_mask(gates["phases"].get("hands", []), F, fps),
        "ankle": _intervals_to_mask(gates["phases"].get("feet", []), F, fps),
    }

    contacts = {}
    for eff, ki in EFFECTOR_KP.items():
        gm = gate_masks["wrist" if "wrist" in eff else "ankle"]
        xy = kp[:, ki, :2]
        conf = kp[:, ki, 2]
        size = np.maximum(bbx[:, 2], 1.0)
        # central-difference pixel speed, bbox-normalized
        vel = np.zeros(F)
        vel[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=1) / 2.0
        vel[0], vel[-1] = vel[1], vel[-2]
        vel = vel / size
        low = xy[:, 1] > bbx[:, 1]        # below person bbox center
        cand = gm & (conf > KP_CONF_MIN) & (vel < KP_SPEED_MAX) & low
        # gap fill
        m = cand.copy()
        idx = np.where(cand)[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= gap + 1:
                m[a:b] = True
        # min duration
        keep = np.zeros(F, dtype=bool)
        i = 0
        while i < F:
            if m[i]:
                j = i
                while j + 1 < F and m[j + 1]:
                    j += 1
                if j - i + 1 >= min_len:
                    keep[i:j + 1] = True
                i = j + 1
            else:
                i += 1
        contacts[eff] = _mask_to_intervals(keep, fps)
    return contacts


def contact_weights(contacts, F, fps):
    """Per-frame per-effector weights in [0,1] with cosine ramps."""
    ramp = max(1, int(round(RAMP_SEC * fps)))
    W = {}
    for eff, intervals in contacts.items():
        w = np.zeros(F)
        for a, b in intervals:
            i0 = max(0, int(round(a * fps)))
            i1 = min(F - 1, int(round(b * fps)))
            for i in range(i0, i1 + 1):
                rise = min(1.0, (i - i0 + 1) / ramp)
                fall = min(1.0, (i1 - i + 1) / ramp)
                w[i] = max(w[i], min(rise, fall))
        W[eff] = w
    return W


# ---------------------------------------------------------------------------
# Stage 3: optimization
# ---------------------------------------------------------------------------

def refine(pkl, gates=None, contacts=None, iters_a=250, iters_b=450, lr=0.01,
           protect_legs=False):
    fps = float(pkl["fps"])
    g = pkl["smpl_params_global"]
    rest = torch.tensor(pkl["mesh"]["joints"], dtype=torch.float64)
    F = g["body_pose"].shape[0]

    # ---- camera
    R_wc, t_wc = fit_static_camera(pkl)
    R_t = torch.tensor(R_wc, dtype=torch.float64)
    t_t = torch.tensor(t_wc, dtype=torch.float64)
    K = torch.tensor(pkl["K_fullimg"], dtype=torch.float64)

    # ---- floor (raw pkl, same recipe as motion_cleanup)
    jw, G_np, pos_np = fk_np(g["body_pose"], g["global_orient"], g["transl"],
                             pkl["mesh"]["joints"].astype(np.float64))
    min_y = lbs_min_height(G_np, pos_np, g["transl"], pkl["mesh"],
                           pkl["mesh"]["joints"].astype(np.float64))
    floor_y = float(np.median(min_y))
    print(f"[floor] estimated floor y = {floor_y:.3f} m")

    # ---- contacts
    if contacts is None:
        contacts = derive_contacts(pkl, gates, fps)
    print("[contacts] derived segments:")
    for eff, iv in contacts.items():
        tot = sum(b - a for a, b in iv)
        print(f"  {eff}: {len(iv)} segments, {tot:.1f}s total  {iv}")
    Wc = contact_weights(contacts, F, fps)
    Wc_t = {e: torch.tensor(w, dtype=torch.float64) for e, w in Wc.items()}

    # ---- data tensors
    bp0 = torch.tensor(g["body_pose"], dtype=torch.float64)
    or0 = torch.tensor(g["global_orient"], dtype=torch.float64)
    tr0 = torch.tensor(g["transl"], dtype=torch.float64)
    kp = torch.tensor(pkl["keypoints_2d"], dtype=torch.float64)
    size = torch.tensor(np.maximum(pkl["bbx_xys"][:, 2], 1.0), dtype=torch.float64)

    smpl_idx = torch.tensor([p[0] for p in REPROJ_PAIRS])
    coco_idx = torch.tensor([p[1] for p in REPROJ_PAIRS])
    kp_xy = kp[:, coco_idx, :2]
    kp_cf = kp[:, coco_idx, 2]
    kp_w = torch.where(kp_cf > KP_CONF_MIN, kp_cf, torch.zeros_like(kp_cf))
    # per-frame body-anchor scale: sharp frames (high mean 2D confidence)
    # loosen the anchor so reprojection can move the pose; blurred frames
    # tighten it so the GVHMR prior holds. Linear: conf 0.9 -> 0.5x, 0.3 -> 1.5x.
    anchor_scale = (2.0 - (5.0 / 3.0) * kp_cf.mean(dim=1)).clamp(0.5, 1.5)

    d_body = torch.zeros_like(bp0, requires_grad=True)
    d_or = torch.zeros_like(or0, requires_grad=True)
    d_tr = torch.zeros_like(tr0, requires_grad=True)

    # leg-protection mask: zero the leg-joint deltas so those joints stay at
    # raw GVHMR (which multi-view paired tests show is already close), while
    # arms/spine/global still refine to plant hands + fix drift.
    body_mask = torch.ones_like(bp0[0])
    if protect_legs:
        idx = np.array([s * 3 + k for s in LEG_BODY_SLOTS for k in range(3)])
        body_mask[idx] = 0.0
        print(f"[protect-legs] freezing {len(LEG_BODY_SLOTS)} leg joints "
              "(hips/knees/ankles/feet) to raw GVHMR")

    def losses(stage_b, sigma):
        body = bp0 + d_body * body_mask if stage_b else bp0
        orient = or0 + d_or
        transl = tr0 + d_tr
        J = fk_t(body, orient, transl, rest)              # (F, 24, 3)

        # reprojection through the constant camera
        Jc = J @ R_t.T + t_t
        Jsel = Jc[:, smpl_idx]
        z = Jsel[..., 2].clamp(min=0.3)
        u = Jsel[..., 0] / z * K[0, 0] + K[0, 2]
        v = Jsel[..., 1] / z * K[1, 1] + K[1, 2]
        r = torch.stack([u, v], dim=-1) - kp_xy
        r = r.norm(dim=-1) / size[:, None]
        gm = r * r / (r * r + sigma * sigma)
        e_reproj = (gm * kp_w).sum() / kp_w.sum()

        # contact clamp + still
        e_contact = J.new_zeros(())
        e_still = J.new_zeros(())
        for eff, tip in EFFECTOR_TIPS.items():
            w = Wc_t[eff]
            if w.sum() < 1:
                continue
            y = J[:, tip, 1]
            target = floor_y + TIP_FLOOR_OFFSET[eff]
            e_contact = e_contact + (w * (y - target) ** 2).sum() / w.sum()
            xz = J[:, tip][:, [0, 2]]
            dxz = (xz[1:] - xz[:-1]).norm(dim=-1) ** 2
            ww = torch.minimum(w[1:], w[:-1])
            if ww.sum() > 0:
                e_still = e_still + (ww * dxz).sum() / ww.sum()

        # smoothness: joint + transl acceleration
        acc = J[2:] - 2 * J[1:-1] + J[:-2]
        e_smooth = (acc.norm(dim=-1) ** 2).mean()

        # anchors (body anchor confidence-scaled per frame)
        e_anchor = (W_ANCHOR_BODY * (anchor_scale[:, None] * d_body ** 2).mean()
                    + W_ANCHOR_ORIENT * (d_or ** 2).mean()
                    + W_ANCHOR_TRANSL * (d_tr ** 2).mean())

        total = (W_REPROJ * e_reproj + W_CONTACT * e_contact
                 + W_STILL * e_still + W_SMOOTH * e_smooth + e_anchor)
        return total, dict(reproj=float(e_reproj), contact=float(e_contact),
                           still=float(e_still), smooth=float(e_smooth),
                           anchor=float(e_anchor))

    def run_stage(params, iters, stage_b, sigma, label):
        opt = torch.optim.Adam(params, lr=lr)
        for it in range(iters):
            opt.zero_grad()
            total, comps = losses(stage_b, sigma)
            total.backward()
            opt.step()
            if it % 50 == 0 or it == iters - 1:
                cs = " ".join(f"{k}={v:.4f}" for k, v in comps.items())
                print(f"[{label} {it:4d}] total={float(total):.4f}  {cs}")

    # stage A: wide robust scale so the large drift errors still pull;
    # stage B: tight scale so blurred-limb outliers stop mattering
    print(f"[opt] stage A: global placement (orient+transl), {iters_a} iters")
    run_stage([d_or, d_tr], iters_a, False, 0.20, "A")
    print(f"[opt] stage B: + body pose, {iters_b} iters")
    run_stage([d_body, d_or, d_tr], iters_b, True, 0.05, "B")

    body = (bp0 + d_body * body_mask).detach().numpy().astype(np.float32)
    orient = (or0 + d_or).detach().numpy().astype(np.float32)
    transl = (tr0 + d_tr).detach().numpy().astype(np.float32)

    # ---- metrics: wrist tip height during hand contacts, before/after
    def tip_heights(bp, orr, tr):
        J, _, _ = fk_np(bp, orr, tr, pkl["mesh"]["joints"].astype(np.float64))
        out = {}
        for eff, tip in EFFECTOR_TIPS.items():
            w = Wc[eff] > 0.5
            out[eff] = float((J[w, tip, 1] - floor_y).mean()) if w.any() else None
        return out

    before = tip_heights(g["body_pose"], g["global_orient"], g["transl"])
    after = tip_heights(body, orient, transl)
    print("[result] mean tip height above floor during contact (before -> after):")
    for eff in EFFECTOR_TIPS:
        b, a = before[eff], after[eff]
        if b is None:
            continue
        print(f"  {eff}: {b*100:6.1f} cm -> {a*100:6.1f} cm")

    report = {
        "contacts": contacts,
        "floor_y": floor_y,
        "R_world2cam": R_wc.tolist(),
        "t_world2cam": t_wc.tolist(),
        "tip_height_before": before,
        "tip_height_after": after,
        "protect_legs": protect_legs,
        "deltas": {
            "transl_max_cm": float(np.abs(d_tr.detach().numpy()).max() * 100),
            "orient_max_deg": float(np.abs(d_or.detach().numpy()).max() * 57.3),
            "body_max_deg": float(np.abs((d_body * body_mask).detach().numpy()).max() * 57.3),
        },
    }
    return body, orient, transl, report


def refine_pkl(input_path, gates_path=None, contacts_path=None, output_path=None,
               iters_a=250, iters_b=450, protect_legs=False):
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        pkl = pickle.load(f)
    if "keypoints_2d" not in pkl:
        raise ValueError(
            f"{input_path} has no keypoints_2d — re-run GVHMR with the "
            "updated gvhmr_modal.py (keypoint export added 2026-07-04)")

    gates = json.loads(Path(gates_path).read_text()) if gates_path else None
    contacts = json.loads(Path(contacts_path).read_text()) if contacts_path else None
    if gates is None and contacts is None:
        raise ValueError("need --gates (coarse windows) or --contacts (explicit)")

    body, orient, transl, report = refine(pkl, gates, contacts, iters_a, iters_b,
                                          protect_legs=protect_legs)

    out = dict(pkl)
    out["smpl_params_global"] = dict(pkl["smpl_params_global"])
    out["smpl_params_global"]["body_pose"] = body
    out["smpl_params_global"]["global_orient"] = orient
    out["smpl_params_global"]["transl"] = transl
    out["refine"] = report

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_refined.pkl")
    output_path = Path(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    cpath = output_path.with_suffix(".contacts.json")
    cpath.write_text(json.dumps(report["contacts"], indent=1))
    print(f"[done] refined pkl: {output_path}")
    print(f"[done] contact segments (inspect/edit): {cpath}")
    return str(output_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="enriched GVHMR pkl (with keypoints_2d)")
    ap.add_argument("--gates", help="coarse phase windows JSON")
    ap.add_argument("--contacts", help="explicit per-effector intervals JSON")
    ap.add_argument("--out", help="output pkl path")
    ap.add_argument("--iters-a", type=int, default=250)
    ap.add_argument("--iters-b", type=int, default=450)
    ap.add_argument("--protect-legs", action="store_true",
                    help="freeze leg joints to raw GVHMR (see leg-extension "
                         "investigation Finding 2)")
    args = ap.parse_args()
    refine_pkl(args.input, args.gates, args.contacts, args.out,
               args.iters_a, args.iters_b, protect_legs=args.protect_legs)


if __name__ == "__main__":
    main()
