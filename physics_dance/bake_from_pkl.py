"""GVHMR clean pkl (smpl_params_global + mesh) -> bake npz (JW, G, rest, fps=30),
the format soma_from_bake.py reads. SMPL's rest pose is an identity T-pose,
so FK's global joint rotations are already "relative to rest".
    python -m physics_dance.bake_from_pkl <clean.pkl>... --out output/physics_dance/bake_ig
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from analyzer_3d.src.motion_cleanup import fk


def to30(JW, G, fps):
    if abs(fps - 30.0) < 1e-3:
        return JW, G
    F = len(JW)
    t_src = np.arange(F) / fps
    t_new = np.arange(0, t_src[-1] + 1e-9, 1 / 30.0)
    JWn = np.stack([np.stack([np.interp(t_new, t_src, JW[:, j, k]) for k in range(3)], -1)
                    for j in range(JW.shape[1])], 1)
    Gn = np.stack([Slerp(t_src, Rotation.from_matrix(G[:, j]))(t_new).as_matrix()
                   for j in range(G.shape[1])], 1)
    return JWn, Gn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkls", nargs="+")
    ap.add_argument("--out", default="output/physics_dance/bake_ig")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f in a.pkls:
        d = pickle.load(open(f, "rb"))
        p = d["smpl_params_global"]
        rest = np.asarray(d["mesh"]["joints"], np.float64)
        jw, G, _ = fk(np.asarray(p["body_pose"]), np.asarray(p["global_orient"]),
                      np.asarray(p["transl"]), rest)
        jw, G = to30(jw, G, float(d["fps"]))
        name = Path(f).name.replace("_smpl_clean.pkl", "")
        np.savez(Path(a.out) / f"{name}.npz", JW=jw, G=G, rest=rest, fps=np.float64(30.0))
        print(name, len(jw), "frames")


if __name__ == "__main__":
    main()
