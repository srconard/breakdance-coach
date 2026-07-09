"""Import BVH/FBX mocap (CMU, Mixamo, ...) straight to a mannequin bundle.

No SMPL involved: skeleton_bake_npz.py samples the source skeleton's own
world transforms per frame (bone-name preset maps them onto SMPL joint
indices purely for the mannequin's part table), then the standard mannequin
GLB builder + viewer packer run on the result.

Usage:
    python -m analyzer_3d.src.mocap_import downloads/cmu85/85_12.bvh --preset cmu
        [--title "CMU 85_12"] [-o output/reference_library/cmu85] [--fps 60]

Presets: cmu (cgspeed/una-dinosauria CMU BVH), mixamo (Adobe Mixamo FBX/BVH).
"""

import argparse
import glob
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .exporter import find_blender
from .make_viewer import make_viewer
from .mannequin import glb_from_npz

BAKE_SCRIPT = Path(__file__).parent / "blender_scripts" / "skeleton_bake_npz.py"


def bake_npz(input_path, out_npz, preset, fps=60.0, blender_path=None):
    blender = blender_path or find_blender()
    if not blender:
        raise FileNotFoundError("Blender not found (needed for mocap bake)")
    cmd = [str(blender), "-b", "--python", str(BAKE_SCRIPT), "--",
           str(Path(input_path).resolve()), str(Path(out_npz).resolve()),
           preset, str(fps)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if "[skeleton_bake] DONE" not in r.stdout:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise RuntimeError(f"skeleton bake failed for {input_path}")
    for line in r.stdout.splitlines():
        if line.startswith("[skeleton_bake]") and "DONE" not in line:
            print(" ", line)
    return out_npz


def import_mocap(input_path, preset, out_dir=None, title=None, fps=60.0,
                 blender_path=None):
    input_path = Path(input_path)
    out_dir = Path(out_dir) if out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    with tempfile.TemporaryDirectory(prefix="mocap_") as td:
        npz = Path(td) / "motion.npz"
        bake_npz(input_path, npz, preset, fps=fps, blender_path=blender_path)
        with np.load(npz) as data:
            out_fps = float(data["fps"])
        glb = glb_from_npz(npz, out_dir / f"{stem}_mannequin.glb", blender_path)
    html = make_viewer(glb, video=None, title=title or stem, fps=out_fps,
                       out=out_dir / f"{stem}_viewer.html")
    return {"mannequin_glb": str(glb), "viewer_html": str(html)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="BVH/FBX file(s) or globs")
    ap.add_argument("--preset", required=True, choices=["cmu", "mixamo"])
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--title", default=None, help="viewer title (single input)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--blender-path", default=None)
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    for p in paths:
        print(f"=== {p}")
        r = import_mocap(p, args.preset, out_dir=args.out_dir,
                         title=args.title if len(paths) == 1 else None,
                         fps=args.fps, blender_path=args.blender_path)
        print(f"  -> {r['viewer_html']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
