"""Reference-free leg-extension metrics — the measuring stick for the
leg-extension accuracy work.

Everything here is computable from a single clip's OWN data (SMPL pose + the
2D keypoints GVHMR already exports). NO reference animation is needed, so the
same numbers can be taken on any raw video of any new move — that is the
signal the refinement method optimizes and the number we watch improve.

Two families:
  GEOMETRY (view-independent 3D angles, degrees):
    knee_angle   L/R interior angle hip-knee-ankle (180 = straight leg)
    straddle     angle between the two thigh directions (leg spread)
    elevation    L/R angle of hip->ankle from world-up (how high the leg kicks)
  We report peak (max over clip) and mean, since under-extension shows up as
  a depressed PEAK (the leg never reaches full extension/height).

  REPROJECTION (the reference-free tuning signal, bbox-normalized px):
    knee/ankle reprojection residual vs the detected 2D keypoint, plus the
    mean keypoint confidence — so we can tell "pose is wrong" (high residual,
    high confidence) from "we can't tell" (low confidence, blurred limb).

FineDance/AIST reference pkls carry no keypoints_2d/K, so only the GEOMETRY
block is computed for them — which is exactly what lets a mocap sequence act
as ground truth for the geometry we're trying to fix.

Usage:
    python -m analyzer_3d.src.leg_metrics motion.pkl
    python -m analyzer_3d.src.leg_metrics raw.pkl refined.pkl   # compare
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from .motion_cleanup import fk as fk_np

# SMPL joints
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 1, 2, 4, 5, 7, 8
UP = np.array([0.0, 1.0, 0.0])  # SMPL world up

# COCO knee/ankle keypoints and their SMPL joints (matches contact_refine)
LEG_REPROJ = {
    "L_knee": (L_KNEE, 13), "R_knee": (R_KNEE, 14),
    "L_ankle": (L_ANK, 15), "R_ankle": (R_ANK, 16),
}
KP_CONF_MIN = 0.35


def _angle(a, b):
    """per-frame angle (deg) between two (F,3) vector arrays."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def geometry(jw):
    """jw (F,24,3) world joints -> dict of angle time-series + peak/mean."""
    knee_L = _angle(jw[:, L_HIP] - jw[:, L_KNEE], jw[:, L_ANK] - jw[:, L_KNEE])
    knee_R = _angle(jw[:, R_HIP] - jw[:, R_KNEE], jw[:, R_ANK] - jw[:, R_KNEE])
    straddle = _angle(jw[:, L_KNEE] - jw[:, L_HIP], jw[:, R_KNEE] - jw[:, R_HIP])
    elev_L = _angle(jw[:, L_ANK] - jw[:, L_HIP], UP[None])
    elev_R = _angle(jw[:, R_ANK] - jw[:, R_HIP], UP[None])
    # elevation above horizontal: 0deg from UP = straight up. report height as
    # (90 - angle_from_up) so bigger = kicked higher; peak = highest kick.
    kick_L = 90.0 - elev_L
    kick_R = 90.0 - elev_R
    out = {}
    for name, ser in [("knee_L", knee_L), ("knee_R", knee_R),
                      ("straddle", straddle), ("kick_L", kick_L),
                      ("kick_R", kick_R)]:
        out[name] = {"peak": float(ser.max()), "mean": float(ser.mean()),
                     "series": ser}
    return out


def reprojection(pkl, jw):
    """knee/ankle reprojection residual + confidence, if the pkl has 2D data.
    Returns None when keypoints/intrinsics are absent (reference datasets)."""
    if "keypoints_2d" not in pkl or "K_fullimg" not in pkl:
        return None
    from .contact_refine import fit_static_camera

    R_wc, t_wc = fit_static_camera(pkl)
    K = np.asarray(pkl["K_fullimg"], dtype=np.float64)
    kp = np.asarray(pkl["keypoints_2d"], dtype=np.float64)
    size = np.maximum(pkl["bbx_xys"][:, 2], 1.0)

    Jc = jw @ R_wc.T + t_wc
    out = {}
    for name, (sj, cj) in LEG_REPROJ.items():
        p = Jc[:, sj]
        z = np.clip(p[:, 2], 0.3, None)
        u = p[:, 0] / z * K[0, 0] + K[0, 2]
        v = p[:, 1] / z * K[1, 1] + K[1, 2]
        proj = np.stack([u, v], axis=-1)
        conf = kp[:, cj, 2]
        resid = np.linalg.norm(proj - kp[:, cj, :2], axis=-1) / size
        hi = conf > KP_CONF_MIN
        out[name] = {
            "resid_hi_conf": float(resid[hi].mean()) if hi.any() else float("nan"),
            "mean_conf": float(conf.mean()),
            "frac_hi_conf": float(hi.mean()),
        }
    return out


def leg_report(pkl_path):
    with open(pkl_path, "rb") as f:
        pkl = pickle.load(f)
    g = pkl["smpl_params_global"]
    jw, _, _ = fk_np(g["body_pose"], g["global_orient"], g["transl"],
                     pkl["mesh"]["joints"].astype(np.float64))
    return {"geometry": geometry(jw), "reproj": reprojection(pkl, jw),
            "frames": jw.shape[0], "fps": float(pkl["fps"])}


def print_report(rep, label=""):
    g = rep["geometry"]
    print(f"\n== leg metrics {label} ({rep['frames']} frames) ==")
    print("  geometry (deg)        peak     mean")
    for k in ("knee_L", "knee_R", "straddle", "kick_L", "kick_R"):
        print(f"    {k:10s}      {g[k]['peak']:7.1f}  {g[k]['mean']:7.1f}")
    if rep["reproj"]:
        print("  reprojection (bbox-norm px, high-conf frames)")
        for k, v in rep["reproj"].items():
            print(f"    {k:10s}  resid={v['resid_hi_conf']:.4f}  "
                  f"conf={v['mean_conf']:.2f}  hi={v['frac_hi_conf']*100:.0f}pct")
    else:
        print("  reprojection: n/a (no 2D keypoints — reference/mocap clip)")


def print_compare(a, b, la="A", lb="B"):
    ga, gb = a["geometry"], b["geometry"]
    print(f"\n== compare {la} -> {lb} ==")
    print("  geometry peak (deg)     {:>8s} {:>8s}    delta".format(la, lb))
    for k in ("knee_L", "knee_R", "straddle", "kick_L", "kick_R"):
        pa, pb = ga[k]["peak"], gb[k]["peak"]
        print(f"    {k:10s}          {pa:8.1f} {pb:8.1f}  {pb - pa:+7.1f}")
    if a["reproj"] and b["reproj"]:
        print("  knee/ankle reproj resid {:>8s} {:>8s}    delta".format(la, lb))
        for k in a["reproj"]:
            ra, rb = a["reproj"][k]["resid_hi_conf"], b["reproj"][k]["resid_hi_conf"]
            print(f"    {k:10s}          {ra:8.4f} {rb:8.4f}  {rb - ra:+7.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkls", nargs="+", help="1 pkl (report) or 2 (compare)")
    args = ap.parse_args()
    reps = [leg_report(p) for p in args.pkls]
    for p, r in zip(args.pkls, reps):
        print_report(r, Path(p).name)
    if len(reps) == 2:
        print_compare(reps[0], reps[1], Path(args.pkls[0]).stem,
                      Path(args.pkls[1]).stem)


if __name__ == "__main__":
    sys.exit(main() or 0)
