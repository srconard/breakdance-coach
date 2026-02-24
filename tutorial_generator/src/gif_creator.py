"""GIF and video clip creator using ffmpeg for extracting and converting video segments."""

import re
import subprocess
from pathlib import Path
from typing import Literal

from tutorial_generator.src.video_analyzer import TutorialStep

OutputFormat = Literal["gif", "mp4", "webm"]


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Args:
        name: The string to sanitize

    Returns:
        A filename-safe string
    """
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r'[<>:"/\\|?*\s]+', '_', name)
    # Remove any leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Lowercase for consistency
    return sanitized.lower()


def create_gif(
    video_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    fps: int = 10,
    width: int = 480,
) -> Path:
    """Create an optimized GIF from a video segment.

    Uses ffmpeg's two-pass palette generation for better quality and smaller size.

    Args:
        video_path: Path to the source video
        output_path: Path for the output GIF
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        fps: Output frame rate
        width: Output width in pixels (height auto-calculated)

    Returns:
        Path to the created GIF
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds

    # Use ffmpeg with palette generation for high-quality, small GIFs
    # Two-pass approach: first generate palette, then create GIF

    palette_path = output_path.parent / f"{output_path.stem}_palette.png"

    # Filter for scaling and fps
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"

    # Pass 1: Generate palette
    palette_cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-ss", str(start_seconds),
        "-t", str(duration),
        "-i", str(video_path),
        "-vf", f"{filters},palettegen=stats_mode=diff",
        "-update", "1",
        str(palette_path),
    ]

    # Pass 2: Create GIF using palette
    gif_cmd = [
        "ffmpeg",
        "-y",  # Overwrite
        "-ss", str(start_seconds),
        "-t", str(duration),
        "-i", str(video_path),
        "-i", str(palette_path),
        "-lavfi", f"{filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5",
        str(output_path),
    ]

    try:
        # Generate palette
        subprocess.run(
            palette_cmd,
            check=True,
            capture_output=True,
        )

        # Create GIF
        subprocess.run(
            gif_cmd,
            check=True,
            capture_output=True,
        )

        # Clean up palette file
        if palette_path.exists():
            palette_path.unlink()

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg error creating GIF: {error_msg}")

    return output_path


def create_video(
    video_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    fps: int = 12,
    width: int = 640,
    format: OutputFormat = "mp4",
) -> Path:
    """Create a video clip from a video segment.

    Args:
        video_path: Path to the source video
        output_path: Path for the output video
        start_seconds: Start time in seconds
        end_seconds: End time in seconds
        fps: Output frame rate
        width: Output width in pixels (height auto-calculated)
        format: Output format ('mp4' or 'webm')

    Returns:
        Path to the created video
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds

    # Build ffmpeg command based on format
    if format == "mp4":
        # H.264 codec for MP4 - widely compatible
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
            "-vf", f"fps={fps},scale={width}:-2",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",  # Enable streaming
            str(output_path),
        ]
    else:  # webm
        # VP9 codec for WebM - smaller files
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_seconds),
            "-t", str(duration),
            "-i", str(video_path),
            "-vf", f"fps={fps},scale={width}:-2",
            "-c:v", "libvpx-vp9",
            "-crf", "30",
            "-b:v", "0",
            "-c:a", "libopus",
            "-b:a", "128k",
            str(output_path),
        ]

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg error creating video: {error_msg}")

    return output_path


def create_clips_for_steps(
    video_path: Path,
    steps: list[TutorialStep],
    output_dir: Path,
    fps: int = 12,
    width: int = 640,
    format: OutputFormat = "gif",
) -> list[Path]:
    """Create clips (GIF or video) for all tutorial steps.

    Args:
        video_path: Path to the source video
        steps: List of tutorial steps
        output_dir: Directory for output files
        fps: Output frame rate
        width: Output width in pixels
        format: Output format ('gif', 'mp4', or 'webm')

    Returns:
        List of paths to created clips
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    format_name = "GIFs" if format == "gif" else f"{format.upper()} videos"

    print(f"\nCreating {format_name} for {len(steps)} steps...")

    for i, step in enumerate(steps, start=1):
        filename = f"step_{i:02d}_{sanitize_filename(step.label)}.{format}"
        output_path = output_dir / filename

        print(f"  [{i}/{len(steps)}] {step.label} ({step.start_time} - {step.end_time})")

        if format == "gif":
            clip_path = create_gif(
                video_path=video_path,
                output_path=output_path,
                start_seconds=step.start_seconds,
                end_seconds=step.end_seconds,
                fps=fps,
                width=width,
            )
        else:
            clip_path = create_video(
                video_path=video_path,
                output_path=output_path,
                start_seconds=step.start_seconds,
                end_seconds=step.end_seconds,
                fps=fps,
                width=width,
                format=format,
            )

        size_kb = clip_path.stat().st_size / 1024
        print(f"       -> {clip_path.name} ({size_kb:.1f} KB)")

        clip_paths.append(clip_path)

    print(f"\nCreated {len(clip_paths)} {format_name} in {output_dir}")

    return clip_paths


def create_gifs_for_steps(
    video_path: Path,
    steps: list[TutorialStep],
    output_dir: Path,
    fps: int = 10,
    width: int = 480,
) -> list[Path]:
    """Create GIFs for all tutorial steps.

    Args:
        video_path: Path to the source video
        steps: List of tutorial steps
        output_dir: Directory for output GIFs
        fps: Output frame rate
        width: Output width in pixels

    Returns:
        List of paths to created GIFs
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_paths = []

    print(f"\nCreating GIFs for {len(steps)} steps...")

    for i, step in enumerate(steps, start=1):
        # Create descriptive filename
        filename = f"step_{i:02d}_{sanitize_filename(step.label)}.gif"
        output_path = output_dir / filename

        print(f"  [{i}/{len(steps)}] {step.label} ({step.start_time} - {step.end_time})")

        gif_path = create_gif(
            video_path=video_path,
            output_path=output_path,
            start_seconds=step.start_seconds,
            end_seconds=step.end_seconds,
            fps=fps,
            width=width,
        )

        # Show file size
        size_kb = gif_path.stat().st_size / 1024
        print(f"       -> {gif_path.name} ({size_kb:.1f} KB)")

        gif_paths.append(gif_path)

    print(f"\nCreated {len(gif_paths)} GIFs in {output_dir}")

    return gif_paths


if __name__ == "__main__":
    # Test the GIF creator
    import sys

    if len(sys.argv) < 4:
        print("Usage: python -m tutorial_generator.src.gif_creator <video_path> <start_seconds> <end_seconds>")
        sys.exit(1)

    video = Path(sys.argv[1])
    start = float(sys.argv[2])
    end = float(sys.argv[3])

    output = create_gif(
        video,
        Path("test_output.gif"),
        start,
        end,
    )
    print(f"Created: {output}")
