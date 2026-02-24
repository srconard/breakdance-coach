"""Re-clip tool for extracting high-quality clips from the original video.

Reads tutorial_metadata.json to get timestamps, then extracts clips from
the original (or any) video at full quality — native fps and resolution.

Usage:
    # Re-clip a specific step at full quality
    python -m src.reclip "output/FLARE_Workout_Exercises" --step 11

    # Re-clip all steps at full quality
    python -m src.reclip "output/FLARE_Workout_Exercises" --all

    # Re-clip from a different source video (e.g. higher quality download)
    python -m src.reclip "output/FLARE_Workout_Exercises" --step 11 --video "hd_video.mp4"

    # Custom output settings
    python -m src.reclip "output/FLARE_Workout_Exercises" --step 11 --fps 30 --width 1280
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_metadata(tutorial_dir: Path) -> dict:
    """Load tutorial metadata from JSON file.

    Args:
        tutorial_dir: Path to the tutorial output directory

    Returns:
        Metadata dictionary

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    metadata_path = tutorial_dir / "tutorial_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata found at {metadata_path}. "
            "This tutorial was created before metadata export was added. "
            "You can regenerate it, or use the timestamps from the markdown file."
        )

    return json.loads(metadata_path.read_text(encoding="utf-8"))


def reclip_step(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_path: Path,
    fps: int | None = None,
    width: int | None = None,
    format: str = "mp4",
) -> Path:
    """Extract a single high-quality clip from a video.

    Args:
        video_path: Path to source video
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        output_path: Path for output clip
        fps: Output fps (None = keep original)
        width: Output width (None = keep original)
        format: Output format (mp4, webm, gif)

    Returns:
        Path to the created clip
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds

    # Build filter chain
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if width:
        filters.append(f"scale={width}:-2")

    if format == "mp4":
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
        ]
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-movflags", "+faststart",
            "-an",
            str(output_path),
        ])
    elif format == "webm":
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
        ]
        if filters:
            cmd.extend(["-vf", ",".join(filters)])
        cmd.extend([
            "-c:v", "libvpx-vp9",
            "-crf", "24",
            "-b:v", "0",
            "-an",
            str(output_path),
        ])
    else:
        raise ValueError(f"Unsupported format for reclip: {format}")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg error:\n{error_msg}")

    return output_path


def download_hq_video(
    source_url: str,
    quality: str = "1080p",
    output_dir: str = "downloads",
    title_hint: str | None = None,
) -> Path:
    """Download a higher quality version of the YouTube video.

    Args:
        source_url: YouTube URL
        quality: Quality preset (720p, 1080p, best)
        output_dir: Directory to save the download
        title_hint: Video title for filename (avoids extra API call)

    Returns:
        Path to the downloaded video
    """
    import re
    from src.downloader import download_video

    # Use quality-tagged filename to avoid overwriting existing downloads
    if title_hint:
        filename = re.sub(r'[<>:"/\\|?*]', '_', title_hint) + f"_{quality}"
    else:
        filename = None  # Let downloader figure it out

    print(f"Downloading {quality} version from YouTube...")
    video_path, title = download_video(
        url=source_url,
        output_dir=output_dir,
        filename=filename,
        quality=quality,
    )
    return video_path


def reclip_from_metadata(
    tutorial_dir: Path,
    step_numbers: list[int] | None = None,
    video_override: Path | None = None,
    download_quality: str | None = None,
    fps: int | None = None,
    width: int | None = None,
    format: str = "mp4",
    output_subdir: str = "hq",
) -> list[Path]:
    """Re-clip steps from a tutorial using stored metadata.

    Args:
        tutorial_dir: Path to the tutorial output directory
        step_numbers: List of step numbers to re-clip (None = all)
        video_override: Use this video instead of the one in metadata
        download_quality: Download a new higher quality video at this quality
            (720p, 1080p, best). Requires source_url in metadata.
        fps: Output fps (None = keep original video fps)
        width: Output width (None = keep original video resolution)
        format: Output format
        output_subdir: Subdirectory name for high-quality clips

    Returns:
        List of paths to created clips
    """
    tutorial_dir = Path(tutorial_dir)
    metadata = load_metadata(tutorial_dir)

    # Find or download the source video
    if download_quality:
        # Download a higher quality version from YouTube
        source_url = metadata.get("source_url")
        if not source_url:
            raise ValueError(
                "No source_url in metadata. Cannot download higher quality. "
                "Add the YouTube URL to tutorial_metadata.json, or use --video."
            )
        video_path = download_hq_video(
            source_url, download_quality,
            title_hint=metadata.get("title"),
        )
    elif video_override:
        video_path = Path(video_override)
    elif metadata.get("original_video"):
        video_path = Path(metadata["original_video"])
    else:
        raise ValueError(
            "No source video found. Provide one with --video, "
            "use --download-hq to fetch from YouTube, "
            "or set 'original_video' in tutorial_metadata.json"
        )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path}\n"
            "The original video may have been moved. "
            "Use --video to specify the current path, "
            "or --download-hq to re-download from YouTube."
        )

    # Check video properties
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_data = json.loads(probe_result.stdout)
    for stream in probe_data.get("streams", []):
        if stream["codec_type"] == "video":
            orig_fps = stream.get("r_frame_rate", "?")
            orig_res = f"{stream.get('width', '?')}x{stream.get('height', '?')}"
            break
    else:
        orig_fps = "?"
        orig_res = "?"

    print(f"Source video: {video_path}")
    print(f"  Resolution: {orig_res}, FPS: {orig_fps}")
    if fps:
        print(f"  Output FPS: {fps}")
    else:
        print(f"  Output FPS: native ({orig_fps})")
    if width:
        print(f"  Output width: {width}px")
    else:
        print(f"  Output width: native")
    print()

    # Filter steps
    steps = metadata["steps"]
    if step_numbers:
        steps = [s for s in steps if s["step_number"] in step_numbers]
        if not steps:
            available = [s["step_number"] for s in metadata["steps"]]
            raise ValueError(
                f"No matching steps found. Available: {available}"
            )

    # Create output directory
    output_dir = tutorial_dir / "gifs" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for step in steps:
        # Build filename
        from src.gif_creator import sanitize_filename
        filename = (
            f"step_{step['step_number']:02d}"
            f"_{sanitize_filename(step['label'])}"
            f"_hq.{format}"
        )
        output_path = output_dir / filename

        print(
            f"  [{step['step_number']}/{len(metadata['steps'])}] "
            f"{step['label']} ({step['start_time']} - {step['end_time']})"
        )

        clip_path = reclip_step(
            video_path=video_path,
            start_seconds=step["start_seconds"],
            end_seconds=step["end_seconds"],
            output_path=output_path,
            fps=fps,
            width=width,
            format=format,
        )

        size_kb = clip_path.stat().st_size / 1024
        print(f"       -> {clip_path.name} ({size_kb:.1f} KB)")
        clip_paths.append(clip_path)

    print(f"\nCreated {len(clip_paths)} high-quality clips in {output_dir}")
    return clip_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-clip tutorial steps at full quality from the original video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-clip step 11 at full quality (native fps & resolution)
  python -m src.reclip "output/FLARE_Workout_Exercises" --step 11

  # Re-clip multiple steps
  python -m src.reclip "output/FLARE_Workout_Exercises" --step 11 --step 16

  # Re-clip all steps at full quality
  python -m src.reclip "output/FLARE_Workout_Exercises" --all

  # Download 1080p from YouTube and re-clip step 11
  python -m src.reclip "output/FLARE_Workout_Exercises" --step 11 --download-hq 1080p

  # Download best available quality and re-clip all
  python -m src.reclip "output/FLARE_Workout_Exercises" --all --download-hq best

  # Use a specific local video file
  python -m src.reclip "output/FLARE_Workout_Exercises" --step 11 --video "hd_video.mp4"

  # Custom fps and width
  python -m src.reclip "output/FLARE_Workout_Exercises" --all --fps 30 --width 1280
        """,
    )

    parser.add_argument(
        "tutorial_dir",
        help="Path to the tutorial output directory",
    )

    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument(
        "--step", "-s",
        type=int,
        action="append",
        dest="steps",
        help="Step number(s) to re-clip (can specify multiple)",
    )
    step_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Re-clip all steps",
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Override source video path",
    )

    parser.add_argument(
        "--download-hq",
        choices=["720p", "1080p", "best"],
        default=None,
        metavar="QUALITY",
        help="Download a higher quality video from YouTube (720p, 1080p, best)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output fps (default: keep original)",
    )

    parser.add_argument(
        "--width", "-w",
        type=int,
        default=None,
        help="Output width in pixels (default: keep original)",
    )

    parser.add_argument(
        "--format", "-f",
        choices=["mp4", "webm"],
        default="mp4",
        help="Output format (default: mp4)",
    )

    parser.add_argument(
        "--output-subdir",
        default="hq",
        help="Subdirectory name for output clips (default: 'hq')",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 50)
    print("HIGH-QUALITY RE-CLIP")
    print("=" * 50)
    print()

    try:
        step_numbers = None if args.all else args.steps

        reclip_from_metadata(
            tutorial_dir=Path(args.tutorial_dir),
            step_numbers=step_numbers,
            video_override=Path(args.video) if args.video else None,
            download_quality=args.download_hq,
            fps=args.fps,
            width=args.width,
            format=args.format,
            output_subdir=args.output_subdir,
        )
        return 0

    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        return 1

    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
