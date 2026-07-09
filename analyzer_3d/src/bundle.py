"""Study-bundle packaging: one pkl → everything Shawn studies from.

Produces, per clip:
    <stem>_mannequin.glb   segmented training mannequin (mannequin.py)
    <stem>_viewer.html     self-contained offline move-study viewer
                           (make_viewer.py; source clip embedded when given)
    <stem>_judgment.json   optional --judge: skinned-GLB preview render →
                           side-by-side vs source → Gemini judge scores

Called automatically by main.py after GLB export; also usable standalone
on any existing pipeline pkl:

    python -m analyzer_3d.src.bundle motion_clean.pkl [--video clip.mp4]
        [--judge] [--title "Flare"] [-o outdir]

The judge stage renders the SKINNED GLB (exporter.py), not the mannequin —
the mannequin GLB's 41 per-object clips don't survive a Blender re-import
(4.4 slotted-actions import gap), and the judge should grade the motion the
same way earlier judgments did.
"""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

from .exporter import find_blender
from .mannequin import export_mannequin
from .make_viewer import make_viewer

RENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "render_preview.py"


def _bundle_stem(pkl_path: Path) -> str:
    return pkl_path.stem.replace("_smpl", "").replace("_clean", "")


def _transcode_for_embed(video, out_path, max_h=720):
    """Shrink the source clip for base64 embedding (viewer panel is small;
    a 1080p60 source would triple the HTML size for nothing). Returns the
    original path unchanged if it is already small."""
    video = Path(video)
    if video.stat().st_size <= 6_000_000:
        return video
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video),
         "-vf", f"scale=-2:'min({max_h},ih)'", "-r", "30",
         "-c:v", "libx264", "-preset", "fast", "-crf", "27", "-an",
         str(out_path)],
        check=True, timeout=600)
    return Path(out_path)


def render_preview(glb, out_mp4, blender_path=None, fps=15, label=None):
    """Render a GLB to a preview MP4 (blender headless, eevee)."""
    blender = blender_path or find_blender()
    if not blender:
        raise FileNotFoundError("Blender not found (needed for preview render)")
    out_mp4 = Path(out_mp4).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(blender), "-b", "--python", str(RENDER_SCRIPT), "--",
           "--input", str(Path(glb).resolve()), "--output", str(out_mp4),
           "--fps", str(fps)]
    if label:
        cmd += ["--label", label]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if not out_mp4.exists():
        print(r.stdout[-3000:])
        print(r.stderr[-2000:])
        raise RuntimeError("preview render failed")
    return out_mp4


def run_judge(source_video, render_glb, out_dir, stem, blender_path=None,
              judge_model=None):
    """Preview render → side-by-side → Gemini judge. Returns judgment dict."""
    from .vlm_judge import DEFAULT_MODEL, build_side_by_side, judge, print_judgment

    out_dir = Path(out_dir)
    preview = out_dir / f"{stem}_preview.mp4"
    print(f"  [judge] rendering preview: {preview.name}")
    render_preview(render_glb, preview, blender_path=blender_path)

    side = out_dir / f"{stem}_vs_render.mp4"
    print(f"  [judge] building side-by-side: {side.name}")
    build_side_by_side(source_video, preview, side)

    j = judge(side, model=judge_model or DEFAULT_MODEL)
    out = out_dir / f"{stem}_judgment.json"
    out.write_text(json.dumps(j, indent=1), encoding="utf-8")
    print(f"  [judge] wrote {out}")
    print_judgment(j)
    return str(out)


def make_bundle(pkl_path, video=None, title=None, out_dir=None, judge=False,
                judge_model=None, skinned_glb=None, blender_path=None,
                video_offset=0.0):
    """Produce the study bundle for one pipeline pkl.

    Returns dict of produced paths: mannequin_glb, viewer_html, and
    (with judge=True) judgment. Judge failures are non-fatal — the bundle
    is still complete without the grade.
    """
    pkl_path = Path(pkl_path)
    out_dir = Path(out_dir) if out_dir else pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _bundle_stem(pkl_path)
    with open(pkl_path, "rb") as f:
        fps = float(pickle.load(f)["fps"])

    glb = export_mannequin(pkl_path, out_dir / f"{stem}_mannequin.glb",
                           blender_path)
    embed = None
    if video:
        embed = _transcode_for_embed(video, out_dir / f"_{stem}_embed_tmp.mp4")
    html = make_viewer(glb, video=embed, title=title or stem.replace("_", " "),
                       fps=fps, video_offset=video_offset,
                       out=out_dir / f"{stem}_viewer.html")
    if embed is not None and embed.name.endswith("_embed_tmp.mp4"):
        embed.unlink(missing_ok=True)
    result = {"mannequin_glb": str(glb), "viewer_html": str(html)}

    if judge:
        if not video:
            print("  [judge] skipped: no source video to compare against")
            return result
        try:
            if skinned_glb is None:
                candidate = out_dir / f"{stem}.glb"
                if candidate.exists():
                    skinned_glb = candidate
                else:
                    from .exporter import export_glb

                    print("  [judge] no skinned GLB found — exporting one")
                    skinned_glb = out_dir / f"{stem}.glb"
                    export_glb(input_pkl=str(pkl_path),
                               output_glb=str(skinned_glb),
                               blender_path=blender_path)
            result["judgment"] = str(
                run_judge(video, skinned_glb, out_dir, stem,
                          blender_path=blender_path, judge_model=judge_model))
        except Exception as e:  # noqa: BLE001 - grading is best-effort
            print(f"  [judge] failed ({type(e).__name__}: {e}) — bundle is "
                  "complete without a grade")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="pipeline pkl (cleaned/refined)")
    ap.add_argument("--video", default=None, help="source clip MP4 to embed + judge against")
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--judge", action="store_true",
                    help="also grade the output with the Gemini judge")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--skinned-glb", default=None,
                    help="existing skinned GLB for the judge render (else auto)")
    ap.add_argument("--video-offset", type=float, default=0.0)
    ap.add_argument("--blender-path", default=None)
    args = ap.parse_args()
    r = make_bundle(args.input, video=args.video, title=args.title,
                    out_dir=args.out_dir, judge=args.judge,
                    judge_model=args.judge_model, skinned_glb=args.skinned_glb,
                    blender_path=args.blender_path,
                    video_offset=args.video_offset)
    print("[bundle] " + json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main() or 0)
