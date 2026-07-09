"""Import AIST++ motion pkls into the pipeline's pkl schema.

AIST++ (https://google.github.io/aistplusplus_dataset/) ships SMPL motion
as smpl_poses (F, 72 axis-angle), smpl_scaling (world units per SMPL unit,
~94 => cm world) and smpl_trans (F, 3), 60 fps, CC BY 4.0. Conversion:

    global_orient = poses[:, :3]
    body_pose     = poses[:, 3:66]        # joints 1..21; hands zeroed by fk
    transl        = trans / scaling       # normalize to SMPL (meter) units

Betas are not distributed (AIST++ optimized a global scale instead), so the
rest skeleton + mesh come from a DONOR pipeline pkl (any GVHMR export). The
donor's proportions are an approximation — fine for reference viewing, not
for millimeter ground truth.

Output pkls drop straight into motion_cleanup.clean_pkl → bundle.make_bundle.

Usage:
    python -m analyzer_3d.src.aist_import downloads/aistpp/motions/gBR_*.pkl
        --donor output/3d/flare60/flare60_demo_smpl_refined_clean.pkl
        -o output/reference_library/aistpp
"""

import argparse
import glob
import pickle
import sys
from pathlib import Path

import numpy as np

AIST_FPS = 60.0


def convert(aist_pkl, donor_mesh, out_pkl):
    with open(aist_pkl, "rb") as f:
        d = pickle.load(f)
    poses = np.asarray(d["smpl_poses"], dtype=np.float32)
    scaling = float(np.asarray(d["smpl_scaling"]).reshape(-1)[0])
    trans = np.asarray(d["smpl_trans"], dtype=np.float32) / scaling
    F = poses.shape[0]

    pkl = {
        "smpl_params_global": {
            "body_pose": poses[:, 3:66].copy(),
            "global_orient": poses[:, :3].copy(),
            "transl": trans,
            "betas": np.zeros((F, 10), dtype=np.float32),
        },
        "fps": AIST_FPS,
        "num_frames": F,
        "mesh": donor_mesh,
        "source": {
            "dataset": "AIST++",
            "sequence": Path(aist_pkl).stem,
            "license": "CC BY 4.0 (motion annotations, Google)",
            "scaling": scaling,
            "note": "rest skeleton/mesh from donor pkl (betas not distributed)",
        },
    }
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(pkl, f)
    print(f"[aist] {Path(aist_pkl).name}: {F} frames ({F / AIST_FPS:.1f}s) -> {out_pkl}")
    return str(out_pkl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="AIST++ motion pkl(s) or globs")
    ap.add_argument("--donor", required=True,
                    help="pipeline pkl providing rest skeleton + mesh")
    ap.add_argument("-o", "--out-dir", default="output/reference_library/aistpp")
    args = ap.parse_args()

    with open(args.donor, "rb") as f:
        donor_mesh = pickle.load(f)["mesh"]

    paths = []
    for pattern in args.inputs:
        hits = sorted(glob.glob(pattern)) or [pattern]
        paths.extend(hits)
    for p in paths:
        convert(p, donor_mesh, Path(args.out_dir) / (Path(p).stem + "_smpl.pkl"))
    print(f"[aist] converted {len(paths)} sequence(s)")


if __name__ == "__main__":
    sys.exit(main() or 0)
