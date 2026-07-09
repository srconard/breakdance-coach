"""Import FineDance motion npys into the pipeline's pkl schema.

FineDance (ICCV 2023, li-ronghui/FineDance) is optical MARKER mocap — real
captured joint angles, unlike AIST++'s multi-view video fits — which makes
its 13 Breaking sequences both reference material and potential ground
truth for the leg-extension work.

Format per sequence npy: (F, 315) float32 at 30 fps
    [:, :3]   root translation (meters)
    [:, 3:]   52 SMPL-H joints x 6D rotation (Zhou et al. continuity rep,
              pytorch3d row-stacking convention)

Conversion: rot6d -> matrix -> axis-angle; keep global(1)+body(21) joints,
drop the 30 hand joints (pipeline fk zeroes SMPL hands anyway). Rest
skeleton/mesh from a donor pipeline pkl (FineDance ships no betas).

Usage:
    python -m analyzer_3d.src.finedance_import downloads/finedance/finedance/motion/052.npy ...
        --donor output/3d/flare60/flare60_demo_smpl_refined_clean.pkl
        -o output/reference_library/finedance [--name-map downloads/finedance/finedance/label_json]
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

FINEDANCE_FPS = 30.0


def rot6d_to_aa(r6):
    """(N, 6) Zhou-et-al 6D -> (N, 3) axis-angle (pytorch3d convention:
    the two 3-vectors are the first two ROWS of R)."""
    a1, a2 = r6[:, :3], r6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-2)  # rows -> (N, 3, 3)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def convert(npy_path, donor_mesh, out_pkl, label=None):
    data = np.load(npy_path)
    F = data.shape[0]
    trans = data[:, :3].astype(np.float32)
    r6 = data[:, 3:].reshape(F, 52, 6)
    aa = rot6d_to_aa(r6[:, :22].reshape(F * 22, 6)).reshape(F, 22, 3)

    pkl = {
        "smpl_params_global": {
            "body_pose": aa[:, 1:22].reshape(F, 63),
            "global_orient": aa[:, 0],
            "transl": trans,
            "betas": np.zeros((F, 10), dtype=np.float32),
        },
        "fps": FINEDANCE_FPS,
        "num_frames": F,
        "mesh": donor_mesh,
        "source": {
            "dataset": "FineDance",
            "sequence": Path(npy_path).stem,
            "label": label,
            "capture": "optical marker mocap",
            "note": "rest skeleton/mesh from donor pkl (no betas distributed); "
                    "hands dropped from SMPL-H",
        },
    }
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(pkl, f)
    print(f"[finedance] {Path(npy_path).name}: {F} frames ({F / FINEDANCE_FPS:.1f}s)"
          f"{' - ' + label['name'] if label else ''} -> {out_pkl}")
    return str(out_pkl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="FineDance motion npy(s) or globs")
    ap.add_argument("--donor", required=True)
    ap.add_argument("-o", "--out-dir", default="output/reference_library/finedance")
    ap.add_argument("--name-map", default=None,
                    help="label_json dir; adds song/style to the pkl + filename")
    args = ap.parse_args()

    with open(args.donor, "rb") as f:
        donor_mesh = pickle.load(f)["mesh"]

    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    for p in paths:
        sid = Path(p).stem
        label = None
        if args.name_map:
            lp = Path(args.name_map) / f"{sid}.json"
            if lp.exists():
                label = json.loads(lp.read_text(encoding="utf-8"))
        suffix = f"_{label['name']}" if label and label.get("name") else ""
        out = Path(args.out_dir) / f"fd{sid}{suffix}_smpl.pkl"
        convert(p, donor_mesh, out, label)
    print(f"[finedance] converted {len(paths)} sequence(s)")


if __name__ == "__main__":
    sys.exit(main() or 0)
