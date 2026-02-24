"""CLI entry point for the 3D Move Analyzer.

Converts breakdancing videos into interactive 3D animated models (GLB)
viewable in Obsidian using the model-viewer plugin.

Pipeline:
    Video → [GVHMR on Modal] → SMPL params → [Blender] → animated .GLB

Usage:
    # Full video → 3D model (GVHMR)
    python -m analyzer_3d.src.main "video.mp4" --backend gvhmr

    # From YouTube URL
    python -m analyzer_3d.src.main "https://youtube.com/watch?v=..." --backend gvhmr

    # Per-step 3D from existing tutorial metadata
    python -m analyzer_3d.src.main "video.mp4" --backend gvhmr \\
        --metadata "output/My_Tutorial/tutorial_metadata.json"

    # Skip GLB export (just get SMPL parameters)
    python -m analyzer_3d.src.main "video.mp4" --backend gvhmr --smpl-only
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3D animated models from breakdancing videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a local video with GVHMR
  python -m analyzer_3d.src.main "video.mp4" --backend gvhmr

  # Analyze a YouTube video
  python -m analyzer_3d.src.main "https://youtube.com/watch?v=..." --backend gvhmr

  # Analyze specific steps from a tutorial
  python -m analyzer_3d.src.main "video.mp4" --backend gvhmr \\
      --metadata "output/My_Tutorial/tutorial_metadata.json" --step 3 --step 5

  # Get SMPL parameters only (skip GLB export)
  python -m analyzer_3d.src.main "video.mp4" --backend gvhmr --smpl-only

Backends:
  gvhmr      - GVHMR on Modal cloud GPU (gravity-aware, best for breakdancing)
               Requires: modal setup + checkpoint upload
               First run: python -m analyzer_3d.src.gvhmr_setup
               Deploy: python -m modal deploy analyzer_3d/src/gvhmr_modal.py
  deepmotion - DeepMotion Animate 3D API (commercial, requires API access)
        """,
    )

    parser.add_argument(
        "input",
        help="Video file path or YouTube URL",
    )

    parser.add_argument(
        "--backend", "-b",
        choices=["gvhmr", "deepmotion"],
        default="gvhmr",
        help="Pose estimation backend (default: gvhmr)",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output directory (default: output/3d/<title>)",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the output (default: derived from filename)",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to tutorial_metadata.json for per-step analysis",
    )

    parser.add_argument(
        "--step", "-s",
        type=int,
        action="append",
        dest="steps",
        help="Step number(s) to analyze (requires --metadata)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override frame rate for GLB animation",
    )

    parser.add_argument(
        "--static-cam",
        action="store_true",
        default=True,
        help="Assume static/tripod camera (default: True)",
    )

    parser.add_argument(
        "--moving-cam",
        action="store_true",
        help="Use moving camera mode (for handheld footage)",
    )

    parser.add_argument(
        "--smpl-only",
        action="store_true",
        help="Only output SMPL parameters (skip GLB export)",
    )

    parser.add_argument(
        "--blender-path",
        type=str,
        default=None,
        help="Path to Blender executable (auto-detected if not set)",
    )

    return parser.parse_args()


def is_youtube_url(s: str) -> bool:
    """Check if a string looks like a YouTube URL."""
    return any(domain in s for domain in ["youtube.com", "youtu.be"])


def get_video_clips(video_path: Path, metadata_path: str, step_numbers: list[int] | None) -> list[dict]:
    """Extract clip info from tutorial metadata.

    Returns list of dicts with: start_seconds, end_seconds, label, clip_path
    """
    import subprocess

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    steps = metadata["steps"]

    if step_numbers:
        steps = [s for s in steps if s["step_number"] in step_numbers]
        if not steps:
            available = [s["step_number"] for s in metadata["steps"]]
            raise ValueError(f"No matching steps. Available: {available}")

    clips = []
    for step in steps:
        # Extract clip from video
        clip_path = video_path.parent / f"_temp_step_{step['step_number']:02d}.mp4"
        duration = step["end_seconds"] - step["start_seconds"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(step["start_seconds"]),
            "-t", str(duration),
            "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-an", str(clip_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        clips.append({
            "path": clip_path,
            "label": step["label"],
            "step_number": step["step_number"],
            "start_seconds": step["start_seconds"],
            "end_seconds": step["end_seconds"],
        })

    return clips


def run_gvhmr(video_path: Path, output_dir: Path, static_cam: bool = True) -> Path:
    """Run GVHMR pose estimation on a video.

    Returns path to the output pickle file.
    """
    from analyzer_3d.src.gvhmr_modal import estimate_pose_gvhmr

    pkl_path = output_dir / f"{video_path.stem}_smpl.pkl"
    estimate_pose_gvhmr(
        input_path=str(video_path),
        output_path=str(pkl_path),
        static_cam=static_cam,
    )
    return pkl_path


def run_export(pkl_path: Path, output_dir: Path, fps: int | None, blender_path: str | None) -> Path:
    """Export SMPL parameters to GLB.

    Returns path to the output GLB file.
    """
    from analyzer_3d.src.exporter import export_glb

    glb_path = output_dir / f"{pkl_path.stem.replace('_smpl', '')}.glb"
    export_glb(
        input_pkl=str(pkl_path),
        output_glb=str(glb_path),
        fps=fps,
        blender_path=blender_path,
    )
    return glb_path


def main() -> int:
    args = parse_args()

    static_cam = not args.moving_cam

    print("=" * 55)
    print("3D MOVE ANALYZER")
    print("=" * 55)

    try:
        # Determine video source
        if is_youtube_url(args.input):
            print("\n[1] Downloading video from YouTube...")
            from shared.downloader import download_video
            temp_dir = Path(tempfile.mkdtemp(prefix="3d_analyzer_"))
            video_path, video_title = download_video(args.input, output_dir=str(temp_dir))
        else:
            video_path = Path(args.input)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            video_title = args.title or video_path.stem

        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            import re
            safe_title = re.sub(r'[<>:"/\\|?*]+', '_', video_title)
            safe_title = re.sub(r'[\s_]+', '_', safe_title).strip('_')[:80]
            output_dir = Path("output") / "3d" / safe_title

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nVideo: {video_path}")
        print(f"Backend: {args.backend}")
        print(f"Output: {output_dir}")

        # Per-step or full video?
        if args.metadata:
            print(f"\n[2] Extracting clips from metadata...")
            clips = get_video_clips(video_path, args.metadata, args.steps)
            print(f"  Found {len(clips)} clips to analyze")

            glb_paths = []
            step_labels = []

            for i, clip in enumerate(clips):
                print(f"\n--- Step {clip['step_number']}: {clip['label']} ---")

                # Run pose estimation
                print(f"[3] Running {args.backend} pose estimation...")
                pkl_path = run_gvhmr(clip["path"], output_dir, static_cam)

                if not args.smpl_only:
                    # Export GLB
                    print(f"[4] Exporting GLB...")
                    glb_path = run_export(pkl_path, output_dir, args.fps, args.blender_path)
                    # Rename with step number
                    final_glb = output_dir / f"step_{clip['step_number']:02d}_{clip['label'].lower().replace(' ', '_')}.glb"
                    glb_path.rename(final_glb)
                    glb_paths.append(final_glb)
                    step_labels.append(clip["label"])

                # Cleanup temp clip
                clip["path"].unlink(missing_ok=True)

        else:
            # Full video analysis
            print(f"\n[2] Running {args.backend} pose estimation...")
            pkl_path = run_gvhmr(video_path, output_dir, static_cam)

            glb_paths = []
            step_labels = [video_title]

            if not args.smpl_only:
                print(f"\n[3] Exporting GLB...")
                glb_path = run_export(pkl_path, output_dir, args.fps, args.blender_path)
                glb_paths.append(glb_path)

        # Generate Obsidian markdown
        if glb_paths and not args.smpl_only:
            print(f"\n[{'5' if args.metadata else '4'}] Generating Obsidian markdown...")
            from analyzer_3d.src.output import generate_3d_markdown

            source_url = args.input if is_youtube_url(args.input) else None
            md_path = generate_3d_markdown(
                title=video_title,
                glb_paths=glb_paths,
                output_dir=output_dir,
                backend=args.backend,
                source_url=source_url,
                step_labels=step_labels,
            )

        print("\n" + "=" * 55)
        print("COMPLETE!")
        print("=" * 55)
        print(f"\nOutput directory: {output_dir}")
        if glb_paths:
            print(f"GLB models: {len(glb_paths)} file(s)")
        if not args.smpl_only:
            print("\nOpen the Markdown file in Obsidian to view your 3D models!")
            print("(Requires the obsidian-model-viewer plugin)")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
