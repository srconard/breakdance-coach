"""Pack the self-contained move-study viewer HTML (quality plan Phase 4).

Inlines the built Three.js bundle + base64 GLB (+ optional source MP4,
+ optional static environment GLB) into one double-clickable HTML file —
offline, no CDN, no server.

Usage:
    python -m analyzer_3d.src.make_viewer model.glb [--video clip.mp4]
        [--scene env.glb] [--title "Flare"] [--fps 30] [--video-offset 0.0]
        [-o out.html]

--scene embeds a STATIC environment (scanned space / photogrammetry mesh /
test set) rendered around the animated mannequin, with a "Scene" UI toggle.
Scale/origin contract (glTF standard): 1 unit = 1 meter, Y-up, ground
surface at y=0, dance spot near the origin — the mannequin animates around
the origin on the y=0 plane. See Docs/scene-glb-and-scan-pipeline.md.

Build the JS bundle once (or after editing viewer/src/app.js):
    cd viewer && npx esbuild src/app.js --bundle --minify --format=iife
        --outfile=dist/app.bundle.js
"""

import argparse
import base64
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEWER_DIR = REPO / "viewer"
BUNDLE = VIEWER_DIR / "dist" / "app.bundle.js"
TEMPLATE = VIEWER_DIR / "template.html"


def ensure_bundle():
    app = VIEWER_DIR / "src" / "app.js"
    if BUNDLE.exists() and BUNDLE.stat().st_mtime >= app.stat().st_mtime:
        return
    print("[viewer] building JS bundle (esbuild)...")
    subprocess.run(
        ["npx", "--yes", "esbuild", "src/app.js", "--bundle", "--minify",
         "--format=iife", "--outfile=dist/app.bundle.js"],
        cwd=VIEWER_DIR, check=True, shell=sys.platform == "win32", timeout=300)


def make_viewer(glb, video=None, title=None, fps=30.0, video_offset=0.0, out=None,
                scene=None):
    glb = Path(glb)
    out = Path(out) if out else glb.with_suffix(".viewer.html")
    title = title or glb.stem.replace("_", " ")
    ensure_bundle()

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("{{TITLE}}", title)
    html = html.replace("{{FPS}}", str(float(fps)))
    html = html.replace("{{VIDEO_OFFSET}}", str(float(video_offset)))
    html = html.replace("{{GLB_B64}}", base64.b64encode(glb.read_bytes()).decode())
    vb64 = base64.b64encode(Path(video).read_bytes()).decode() if video else ""
    html = html.replace("{{VIDEO_B64}}", vb64)
    sb64 = base64.b64encode(Path(scene).read_bytes()).decode() if scene else ""
    html = html.replace("{{SCENE_B64}}", sb64)
    html = html.replace("{{BUNDLE_JS}}", BUNDLE.read_text(encoding="utf-8"))

    out.write_text(html, encoding="utf-8")
    print(f"[viewer] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return str(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("glb")
    ap.add_argument("--video", default=None, help="source clip MP4 for side-by-side sync")
    ap.add_argument("--scene", default=None,
                    help="static environment GLB (meters, Y-up, ground at y=0, "
                         "dance spot near origin)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--video-offset", type=float, default=0.0,
                    help="seconds: video time = model time + offset")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    make_viewer(args.glb, args.video, args.title, args.fps, args.video_offset, args.out,
                scene=args.scene)


if __name__ == "__main__":
    main()
