"""Experiment 1 — which clips could a physics body copy at all? ($0, CPU)

Kinematic physics-plausibility screen (in the spirit of KungfuBot's filter):
for each clip, pose the body, find its floor contacts and centre of mass,
and measure what a simulated body could not do:

  penetration   body below the floor (cm, % frames)
  floating      no floor contact for a stretch whose centre of mass does
                not fall like a thrown object (hover, not a jump)
  sliding       contact points skating along the floor (m/s)
  balance       in slow (quasi-static) frames, the centre of mass outside
                the support area of the contacts
  jerk          centre-of-mass accelerations no body produces (> 3 g),
                root teleports, joint spin rates > 30 rad/s

Verdict per clip: trackable-now / needs-cleaning / broken, with the reasons.
Inputs: GVHMR clean pkls (smpl_params_global + mesh) or skeleton bakes (npz).
    python -m physics_dance.feasibility output/ig_ingest/results/*/*_smpl_clean.pkl \
        output/physics_dance/bake/mixamo_fw1.npz --out output/physics_dance/feasibility
"""
import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np

from analyzer_3d.src.motion_cleanup import fk, savgol

G_ACC = 9.81
# approximate segment mass fractions per SMPL joint (de Leva 1996, folded)
MASS = np.array([0.142, 0.1416, 0.1416, 0.10, 0.0433, 0.0433, 0.10, 0.0107,
                 0.0107, 0.10, 0.003, 0.003, 0.02, 0.01, 0.01, 0.06, 0.0271,
                 0.0271, 0.0162, 0.0162, 0.0061, 0.0061, 0.0, 0.0])
MASS = MASS / MASS.sum()
# clearance of a joint centre above the floor when that part touches it (bakes)
JOINT_CLEAR = {0: 0.10, 4: 0.06, 5: 0.06, 7: 0.075, 8: 0.075, 10: 0.03, 11: 0.03,
               15: 0.10, 16: 0.05, 17: 0.05, 18: 0.04, 19: 0.04, 20: 0.035, 21: 0.035,
               12: 0.06, 9: 0.10, 6: 0.10, 3: 0.10}
CONTACT_H = 0.03  # m


def _segments_com(points, owner):
    """COM from per-point owner joint: mass of joint j spread over its points."""
    com = np.zeros((points.shape[0], 3))
    for j in range(24):
        sel = owner == j
        if sel.any() and MASS[j] > 0:
            com += MASS[j] * points[:, sel].mean(1)
    return com / MASS[[j for j in range(24) if (owner == j).any()]].sum()


def load_pkl(path, stride=4):
    d = pickle.load(open(path, "rb"))
    p = d["smpl_params_global"]
    mesh = d["mesh"]
    rest_j = np.asarray(mesh["joints"], np.float64)
    jw, G, pos = fk(np.asarray(p["body_pose"]), np.asarray(p["global_orient"]),
                    np.asarray(p["transl"]), rest_j)
    V = np.asarray(mesh["vertices"], np.float64)[::stride]
    W = np.asarray(mesh["weights"], np.float64)[::stride]
    tj = pos - np.einsum("fjab,jb->fja", G, rest_j)
    verts = np.einsum("vj,fjab,vb->fva", W, G, V, optimize=True) + np.einsum("vj,fja->fva", W, tj) \
        + np.asarray(p["transl"])[:, None]
    owner = W.argmax(1)
    return {"points": verts, "clear": np.zeros(len(V)), "owner": owner,
            "root": jw[:, 0], "G": G, "fps": float(d["fps"]), "kind": "mesh"}


def load_bake(path):
    d = np.load(path)
    J = d["JW"].astype(np.float64)
    G = d["G"].astype(np.float64)
    idx = sorted(JOINT_CLEAR)
    clear = np.array([JOINT_CLEAR[j] for j in idx])
    return {"points": J[:, idx], "clear": clear, "owner": np.array(idx),
            "root": J[:, 0], "G": G, "fps": float(d["fps"]), "kind": "joints",
            "com_points": J}


def runs(mask):
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i))
            s = None
    return out


def hull_contains(pts, q, margin):
    """2-D: is q inside the convex hull of pts grown by margin?"""
    if len(pts) == 0:
        return False
    if len(pts) < 3:
        return np.min(np.linalg.norm(pts - q, axis=1)) <= margin + 0.05
    P = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(P):
        h = []
        for p in P:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h
    H = np.array(half(P)[:-1] + half(P[::-1])[:-1])
    if len(H) < 3:
        return np.min(np.linalg.norm(pts - q, axis=1)) <= margin + 0.05
    for a, b in zip(H, np.roll(H, -1, 0)):
        e = b - a
        n = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-9)
        if np.dot(q - a, n) > margin:
            return False
    return True


def analyse(c):
    P, fps = c["points"], c["fps"]
    F = P.shape[0]
    h = P[..., 1] - c["clear"][None]  # height of each point's surface
    if c["kind"] == "mesh":
        com = _segments_com(P, c["owner"])
    else:
        J = c["com_points"]
        com = (MASS[None, :, None] * J).sum(1)
    w = max(5, int(round(0.2 * fps)) | 1)
    com_s = savgol(com, w, 3, axis=0)
    vel = np.gradient(com_s, 1 / fps, axis=0)
    acc = np.gradient(vel, 1 / fps, axis=0)
    minh = h.min(1)
    contact = h < CONTACT_H
    airborne = ~contact.any(1)
    # floating: airborne > 0.2 s whose COM vertical accel is not ~ -g
    float_s, jumps = 0.0, 0
    for a, b in runs(airborne):
        if (b - a) / fps < 0.2:
            continue
        ay = np.polyfit(np.arange(a, b) / fps, com[a:b, 1], 2)[0] * 2
        if ay > -0.5 * G_ACC:
            float_s += (b - a) / fps
        else:
            jumps += 1
    # sliding: horizontal speed of points that stay in contact
    both = contact[1:] & contact[:-1]
    dxz = np.linalg.norm(P[1:, :, [0, 2]] - P[:-1, :, [0, 2]], axis=-1) * fps
    slide = np.array([np.median(dxz[i][both[i]]) for i in range(F - 1) if both[i].any()])
    slide_p90 = float(np.percentile(slide, 90)) if len(slide) else 0.0
    # balance in quasi-static frames
    quasi = (np.linalg.norm(acc[:, [0, 2]], axis=1) < 1.5) & contact.any(1)
    outside = 0
    for i in np.nonzero(quasi)[0]:
        if not hull_contains(P[i][contact[i]][:, [0, 2]], com[i, [0, 2]], 0.05):
            outside += 1
    bal_out = 100.0 * outside / max(1, quasi.sum())
    # jerk / teleports / spin rates
    acc_p99 = float(np.percentile(np.linalg.norm(acc, axis=1), 99))
    jump_root = float(np.max(np.linalg.norm(np.diff(c["root"], axis=0), axis=1)))
    Gm = c["G"]
    rel = np.einsum("fjba,fjbc->fjac", Gm[:-1], Gm[1:])
    ang = np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)) * fps
    spin_p99 = float(np.percentile(ang, 99))
    m = {
        "seconds": round(F / fps, 1),
        "penetration_max_cm": round(float(max(0.0, -minh.min()) * 100), 1),
        "penetration_pct": round(100 * float((minh < -0.03).mean()), 1),
        "contact_pct": round(100 * float(contact.any(1).mean()), 1),
        "floating_s": round(float_s, 2), "jumps": jumps,
        "slide_p90_mps": round(slide_p90, 2),
        "balance_out_pct": round(bal_out, 1), "quasi_static_frames": int(quasi.sum()),
        "com_acc_p99_g": round(acc_p99 / G_ACC, 2),
        "root_jump_max_m": round(jump_root, 3),
        "spin_p99_rad_s": round(spin_p99, 1),
    }
    broken, clean = [], []
    if m["floating_s"] > 0.5:
        broken.append(f"floats {m['floating_s']} s with no support")
    if m["com_acc_p99_g"] > 6 or m["root_jump_max_m"] > 0.25:
        broken.append(f"impossible jolts (COM {m['com_acc_p99_g']} g, root jump {m['root_jump_max_m']} m)")
    if m["penetration_pct"] > 10 and m["penetration_max_cm"] > 15:
        broken.append(f"sinks {m['penetration_max_cm']} cm into the floor")
    if 0.2 < m["floating_s"] <= 0.5:
        clean.append(f"hovers {m['floating_s']} s")
    if m["slide_p90_mps"] > 0.3:
        clean.append(f"contacts skate ({m['slide_p90_mps']} m/s)")
    if m["balance_out_pct"] > 25 and m["quasi_static_frames"] > 10:
        clean.append(f"off balance in {m['balance_out_pct']}% of slow frames")
    if 3 < m["com_acc_p99_g"] <= 6:
        clean.append(f"jerky ({m['com_acc_p99_g']} g peaks)")
    if m["spin_p99_rad_s"] > 30:
        clean.append(f"joint spin spikes ({m['spin_p99_rad_s']} rad/s)")
    if m["penetration_pct"] > 5:
        clean.append(f"through the floor in {m['penetration_pct']}% of frames")
    m["verdict"] = "broken" if broken else ("needs-cleaning" if clean else "trackable-now")
    m["reasons"] = broken + clean
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="output/physics_dance/feasibility")
    ap.add_argument("--tags", default="output/ig_ingest/tags.json")
    a = ap.parse_args()
    files = [f for g in a.inputs for f in sorted(glob.glob(g))]
    tags = {}
    try:
        tags = json.load(open(a.tags, encoding="utf-8"))
    except Exception:
        pass
    rows = []
    for f in files:
        name = Path(f).name.replace("_smpl_clean.pkl", "").replace(".npz", "")
        try:
            c = load_pkl(f) if f.endswith(".pkl") else load_bake(f)
            m = analyse(c)
        except Exception as ex:
            m = {"verdict": "error", "reasons": [repr(ex)]}
        t = tags.get(name) if isinstance(tags, dict) else None
        if isinstance(t, dict):
            m["move"] = t.get("move") or t.get("primary_move") or t.get("moves") or t.get("label")
        m["clip"] = name
        rows.append(m)
        print(name, m["verdict"], "; ".join(m["reasons"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(Path(a.out) / "feasibility.json", "w"), indent=1, default=str)
    from collections import Counter
    print("SUMMARY", dict(Counter(r["verdict"] for r in rows)))


if __name__ == "__main__":
    main()
