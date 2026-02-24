"""Frame interpolation tool for creating slow-motion breakdance tutorial clips.

Uses FFmpeg's minterpolate filter (optical flow) to generate smooth slow-motion
videos from existing tutorial clips. Designed as a standalone tool that Claude
can run on-demand when the user wants to study a specific move in detail.

Usage:
    python -m src.interpolate "clip.mp4" --slowdown 3
    python -m src.interpolate "clip.mp4" --slowdown 4 --fps 60 -o "slowmo.mp4"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def interpolate_video(
    input_path: Path,
    output_path: Path | None = None,
    slowdown: float = 2.0,
    output_fps: int = 60,
    mi_mode: str = "mci",
    mc_mode: str = "aobmc",
) -> Path:
    """Create a slow-motion version of a video using frame interpolation.

    Uses FFmpeg's minterpolate filter with motion-compensated interpolation
    for smooth slow-motion output.

    Args:
        input_path: Path to the input video file
        output_path: Path for the output video (auto-generated if None)
        slowdown: Slow-motion factor (2 = half speed, 3 = third speed, etc.)
        output_fps: Output frame rate (default: 60fps for smooth playback)
        mi_mode: Motion interpolation mode ('mci' for motion compensated)
        mc_mode: Motion compensation mode ('aobmc' for adaptive overlapped block)

    Returns:
        Path to the created slow-motion video
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Auto-generate output path if not provided
    if output_path is None:
        suffix = input_path.suffix or ".mp4"
        output_path = input_path.with_name(
            f"{input_path.stem}_slowmo_{slowdown}x{suffix}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate the interpolated fps needed before slowdown
    # To get 60fps output at 3x slowdown, we need 180fps interpolated frames
    interpolated_fps = int(output_fps * slowdown)

    print(f"Frame Interpolation Settings:")
    print(f"  Input: {input_path}")
    print(f"  Slowdown: {slowdown}x")
    print(f"  Interpolating to {interpolated_fps}fps, then slowing to {output_fps}fps")
    print(f"  Mode: {mi_mode} / {mc_mode}")
    print(f"  Output: {output_path}")
    print()

    # Build the filter chain:
    # 1. minterpolate: generate interpolated frames at high fps
    # 2. setpts: slow down by multiplying presentation timestamps
    vf_filter = (
        f"minterpolate=fps={interpolated_fps}:mi_mode={mi_mode}:mc_mode={mc_mode},"
        f"setpts={slowdown}*PTS"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", vf_filter,
        "-r", str(output_fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",  # Higher quality for slow-mo study
        "-movflags", "+faststart",
        "-an",  # Remove audio (meaningless when slowed down)
        str(output_path),
    ]

    print("Processing... (this may take a moment for longer clips)")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg interpolation error:\n{error_msg}")

    # Report results
    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\nDone!")
    print(f"  Input:  {input_size:.1f} MB")
    print(f"  Output: {output_size:.1f} MB")
    print(f"  Slow-motion: {slowdown}x at {output_fps}fps")
    print(f"  File: {output_path}")

    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create slow-motion videos using frame interpolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.interpolate "clip.mp4" --slowdown 2
  python -m src.interpolate "step_05.mp4" --slowdown 4 --fps 60
  python -m src.interpolate "clip.mp4" --slowdown 3 -o "slowmo_clip.mp4"

Slowdown guide:
  2x  - Good for most moves, easy to follow
  3x  - Great for fast transitions and footwork
  4x  - Detailed study of rapid movements (flares, windmills)
  5x+ - Frame-by-frame analysis of very fast moves
        """,
    )

    parser.add_argument(
        "input",
        help="Path to the input video file",
    )

    parser.add_argument(
        "--slowdown", "-s",
        type=float,
        default=2.0,
        help="Slow-motion factor (default: 2 = half speed)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Output frame rate (default: 60)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )

    parser.add_argument(
        "--mi-mode",
        choices=["dup", "blend", "mci"],
        default="mci",
        help="Interpolation mode: dup (duplicate), blend (average), mci (motion compensated, default)",
    )

    parser.add_argument(
        "--mc-mode",
        choices=["obmc", "aobmc"],
        default="aobmc",
        help="Motion compensation: obmc (overlapped block), aobmc (adaptive, default)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print("=" * 50)
    print("FRAME INTERPOLATION - SLOW MOTION")
    print("=" * 50)
    print()

    try:
        output_path = args.output
        if output_path:
            output_path = Path(output_path)

        interpolate_video(
            input_path=Path(args.input),
            output_path=output_path,
            slowdown=args.slowdown,
            output_fps=args.fps,
            mi_mode=args.mi_mode,
            mc_mode=args.mc_mode,
        )
        return 0

    except FileNotFoundError as e:
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
