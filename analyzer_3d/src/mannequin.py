"""Segmented training-mannequin GLB exporter (quality plan Phase 2).

Instead of skinning the SMPL mesh (LBS artifacts are worst exactly on
breaking poses), this builds a rigid part per bone — cylinders between
joints, ball joints at articulations — and bakes each part's WORLD
transform per frame from SMPL forward kinematics. Zero skinning artifacts
by construction, tiny files, deliberate training-dummy aesthetic with
left/right color coding (left teal, right orange).

Usage:
    python -m analyzer_3d.src.mannequin input_clean.pkl [-o out.glb]
"""

import argparse
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .exporter import find_blender
from .motion_cleanup import fk

BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "mannequin_glb.py"


def glb_from_npz(npz_path, output_glb, blender_path=None):
    """Run the Blender mannequin builder on an existing motion npz
    (JW/G/rest/fps — see mannequin_glb.py). Used by export_mannequin and by
    the BVH/FBX import path (mocap_import.py), which has no SMPL pkl."""
    blender = blender_path or find_blender()
    if not blender:
        raise FileNotFoundError("Blender not found (needed for GLB export)")
    output_glb = Path(output_glb)
    cmd = [str(blender), "-b", "--python", str(BLENDER_SCRIPT), "--",
           str(npz_path), str(output_glb.resolve())]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    marker = "[mannequin_glb] DONE"
    if marker not in r.stdout:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise RuntimeError("mannequin Blender export failed")
    for line in r.stdout.splitlines():
        if line.startswith("[mannequin_glb]"):
            print(" ", line)
    print(f"[mannequin] wrote {output_glb} ({output_glb.stat().st_size / 1e6:.2f} MB)")
    return str(output_glb)


def export_mannequin(input_pkl, output_glb=None, blender_path=None):
    input_pkl = Path(input_pkl)
    if output_glb is None:
        output_glb = input_pkl.with_name(
            input_pkl.stem.replace("_smpl", "").replace("_clean", "") + "_mannequin.glb")
    output_glb = Path(output_glb)

    with open(input_pkl, "rb") as f:
        pkl = pickle.load(f)
    g = pkl["smpl_params_global"]
    rest = pkl["mesh"]["joints"].astype(np.float64)
    jw, G, _ = fk(g["body_pose"], g["global_orient"], g["transl"], rest)
    fps = float(pkl["fps"])
    print(f"[mannequin] {jw.shape[0]} frames @ {fps:.2f} fps")

    with tempfile.TemporaryDirectory(prefix="mannequin_") as td:
        npz = Path(td) / "motion.npz"
        np.savez(npz, JW=jw, G=G, rest=rest, fps=np.float64(fps))
        return glb_from_npz(npz, output_glb, blender_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="cleaned/refined SMPL pkl")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--blender-path", default=None)
    args = ap.parse_args()
    export_mannequin(args.input, args.out, args.blender_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
