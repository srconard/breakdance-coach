"""Sanity-check a bake npz against what soma_from_bake.py assumes (2026-11-24).

soma_from_bake.py copies the bake's G (each bone's world rotation relative to
its own REST) straight onto SOMA23, whose zero pose is a T-pose facing +Z with
its left at +X (in the bake's Y-up frame). That is only right when the bake's
rest skeleton is that same T-pose. This prints, per npz: the rest facing, the
rest arm and leg directions, and frame 0's facing, so a wrong rest shows up
before any Modal money is spent.

    python -m physics_dance.check_bake_rest a.npz [b.npz ...]
"""
import sys

import numpy as np


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def report(path):
    d = np.load(path)
    rest, JW, G = d["rest"], d["JW"], d["G"]
    left = unit(rest[1] - rest[2])           # right hip -> left hip
    up = unit(rest[12] - rest[0])            # hips -> neck
    face = unit(np.cross(left, up))          # +X x +Y = +Z
    larm = unit(rest[20] - rest[16])          # left shoulder(arm) -> left wrist
    rarm = unit(rest[21] - rest[17])
    lleg = unit(rest[7] - rest[1])
    f0_left = unit(JW[0, 1] - JW[0, 2])
    f0_up = unit(JW[0, 12] - JW[0, 0])
    np.set_printoptions(precision=2, suppress=True)
    print(f"== {path}  frames={len(JW)} fps={float(d['fps']):.1f}")
    print("  rest left(hipR->hipL)", left, " up(hips->neck)", up, " facing", face)
    print("  rest L arm dir", larm, " R arm dir", rarm, " L leg dir", lleg)
    print("  frame0 left", f0_left, " up", f0_up)
    print("  G[0] root angle deg", np.degrees(np.arccos(np.clip((np.trace(G[0, 0]) - 1) / 2, -1, 1))).round(1))
    ang = np.degrees(np.arccos(np.clip((np.einsum("fjii->fj", G) - 1) / 2, -1, 1)))  # (F, 24)
    step = np.degrees(np.arccos(np.clip((np.einsum("fjik,fjik->fj", G[1:], G[:-1]) - 1) / 2, -1, 1)))
    jump = np.linalg.norm(JW[1:, 0] - JW[:-1, 0], axis=1)
    print("  mean bone angle from rest, frames 0..5:", ang[:6].mean(1).round(1))
    print("  max per-frame bone turn, frames 0..5 (deg):", step[:5].max(1).round(1),
          " clip median", float(np.median(step.max(1))).__round__(1))
    print("  root move per frame 0..5 (m):", jump[:5].round(3), " clip median", round(float(np.median(jump)), 3))
    print("  root height frame0 / clip median (m):", round(float(JW[0, 0, 1]), 2), round(float(np.median(JW[:, 0, 1])), 2))
    ok = (left[0] > 0.9 and up[1] > 0.9 and abs(larm[0]) > 0.9 and lleg[1] < -0.9)
    print("  T-pose facing +Z, left +X:", "YES" if ok and face[2] > 0.9 else "NO")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
