"""Score baked clips (skeleton_bake_npz output) for the six-step signature.

A six-step: hands (one or both) stay on the floor near a fixed spot while the
feet travel a full circle around them in ~6 steps. We measure, Y-up:
  - wrist / ankle / pelvis heights (is it floorwork?),
  - the unwrapped angle of the ankle-midpoint around the support-hand centroid
    (one six-step cycle = ~360 deg),
  - the hip-yaw drift.
Usage: python -m physics_dance.inspect_sixstep output/physics_dance/bake/*.npz
"""
import glob
import sys

import numpy as np

PELVIS, L_ANK, R_ANK, L_WR, R_WR, HEAD = 0, 7, 8, 20, 21, 15


def score(path):
    d = np.load(path)
    J = d["JW"]  # (F,24,3) y-up metres
    fps = float(d["fps"])
    F = len(J)
    wr = J[:, [L_WR, R_WR]]
    ank = J[:, [L_ANK, R_ANK]]
    low_hand = wr[:, :, 1].min(1)
    pel_h = J[:, PELVIS, 1]
    # support centre: the lower wrist when it is within 15 cm of the floor
    sup = np.where((wr[:, 0, 1] < wr[:, 1, 1])[:, None], wr[:, 0], wr[:, 1])
    mid_ank = ank.mean(1)
    v = mid_ank[:, [0, 2]] - sup[:, [0, 2]]
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    floor = low_hand < 0.15
    out = {
        "file": path.replace("\\", "/").split("/")[-1],
        "frames": F, "sec": round(F / fps, 2),
        "hand_on_floor_pct": round(100 * floor.mean(), 1),
        "pelvis_h_med": round(float(np.median(pel_h)), 2),
        "feet_turn_deg_total": round(float(np.degrees(ang[-1] - ang[0])), 0),
    }
    # longest run with a hand down, and the turn of the feet during it
    best, cur, start = (0, 0, 0), 0, 0
    for i, f in enumerate(list(floor) + [False]):
        if f:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[0]:
                best = (cur, start, i)
            cur = 0
    n, s, e = best
    if n > 1:
        out["floor_run_s"] = [round(s / fps, 2), round(e / fps, 2)]
        out["feet_turn_deg_in_run"] = round(float(np.degrees(ang[e - 1] - ang[s])), 0)
    return out


if __name__ == "__main__":
    png = "--png" in sys.argv
    files = [f for a in sys.argv[1:] if a != "--png" for f in glob.glob(a)]
    for f in files:
        print(score(f))
        if png:
            sheet(f, f[:-4] + "_sheet.png")


BONES = [(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11),
         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (13, 16),
         (16, 18), (18, 20), (9, 14), (14, 17), (17, 19), (19, 21)]


def sheet(path, out_png, n=12):
    """Contact sheet: n poses side view (x,y) + a top-view feet/hands trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(path)
    J = d["JW"]
    F = len(J)
    idx = np.linspace(0, F - 1, n).astype(int)
    fig, axes = plt.subplots(2, n // 2 + 1, figsize=(3 * (n // 2 + 1), 6))
    axes = axes.ravel()
    for k, i in enumerate(idx):
        ax = axes[k]
        for a, b in BONES:
            c = "tab:red" if a in (1, 4, 7, 13, 16, 18) or b in (1, 4, 7, 10, 13, 16, 18, 20) else "tab:blue"
            ax.plot(J[i, [a, b], 0], J[i, [a, b], 1], c=c, lw=2)
        ax.axhline(0, c="k", lw=0.5)
        ax.set_xlim(J[:, 0, 0].mean() - 1, J[:, 0, 0].mean() + 1)
        ax.set_ylim(-0.05, 1.8)
        ax.set_aspect("equal")
        ax.set_title(f"t={i / float(d['fps']):.2f}s", fontsize=8)
        ax.axis("off")
    ax = axes[len(idx)]
    for j, c in ((7, "tab:red"), (8, "tab:blue"), (20, "salmon"), (21, "lightblue")):
        ax.plot(J[:, j, 0], J[:, j, 2], c=c, lw=1)
    ax.set_aspect("equal")
    ax.set_title("top view: feet(dark) hands(light)", fontsize=8)
    for ax in axes[len(idx) + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=60)
