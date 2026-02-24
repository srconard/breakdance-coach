"""Frame interpolation tool for creating slow-motion breakdance tutorial clips.

Supports multiple backends:
  - ffmpeg:  FFmpeg minterpolate (basic, no GPU needed, artifacts on fast motion)
  - rife:    RIFE v4.25 on Modal cloud GPU (high quality, requires Modal setup)

Usage:
    python -m src.interpolate "clip.mp4" --slowdown 3
    python -m src.interpolate "clip.mp4" --slowdown 3 --backend rife
    python -m src.interpolate "clip.mp4" --slowdown 4 --fps 60 -o "slowmo.mp4"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def interpolate_ffmpeg(
    input_path: Path,
    output_path: Path,
    slowdown: float = 2.0,
    output_fps: int = 60,
    mi_mode: str = "mci",
    mc_mode: str = "aobmc",
) -> Path:
    """Interpolate using FFmpeg minterpolate filter (basic quality)."""
    interpolated_fps = int(output_fps * slowdown)

    print(f"Backend: FFmpeg minterpolate")
    print(f"  Interpolating to {interpolated_fps}fps, then slowing to {output_fps}fps")
    print(f"  Mode: {mi_mode} / {mc_mode}")
    print()

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
        "-crf", "18",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    print("Processing... (this may take a moment for longer clips)")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg interpolation error:\n{error_msg}")

    return output_path


def interpolate_rife(
    input_path: Path,
    output_path: Path,
    slowdown: float = 2.0,
    output_fps: int = 60,
) -> Path:
    """Interpolate using RIFE v4.25 on Modal cloud GPU (high quality).

    For slowdown factors > 2, uses iterative 2x passes for better quality
    on fast motion (less ghosting than a single large multiplier).
    """
    from src.rife_modal import interpolate_video_rife

    multi = int(slowdown)
    if multi != slowdown:
        print(f"  Note: RIFE requires integer multiplier, rounding {slowdown} to {multi}")

    # For multi > 2, run iterative 2x passes for better quality
    if multi > 2 and (multi & (multi - 1)) == 0:
        # Power of 2 (4, 8, etc.) — do iterative 2x
        passes = 0
        m = multi
        while m > 1:
            m //= 2
            passes += 1
        print(f"  Strategy: {passes} iterative 2x passes (better quality than single {multi}x)")

        current_input = input_path
        for i in range(passes):
            pass_output = output_path.with_name(
                f"{output_path.stem}_pass{i+1}{output_path.suffix}"
            ) if i < passes - 1 else output_path

            print(f"\n  Pass {i+1}/{passes}:")
            interpolate_video_rife(
                input_path=str(current_input),
                output_path=str(pass_output),
                multi=2,
            )
            # Clean up intermediate files
            if i > 0 and current_input != input_path:
                current_input.unlink(missing_ok=True)
            current_input = pass_output
    else:
        # Single pass for non-power-of-2 or small multipliers
        interpolate_video_rife(
            input_path=str(input_path),
            output_path=str(output_path),
            multi=multi,
        )

    # If the user wants a specific output FPS different from just multi * input_fps,
    # re-encode with the target fps
    if output_fps != 60:
        temp_path = output_path.with_name(f"{output_path.stem}_temp{output_path.suffix}")
        output_path.rename(temp_path)
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(temp_path),
                "-r", str(output_fps),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-movflags", "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
        temp_path.unlink(missing_ok=True)

    return output_path


def interpolate_video(
    input_path: Path,
    output_path: Path | None = None,
    slowdown: float = 2.0,
    output_fps: int = 60,
    backend: str = "ffmpeg",
    mi_mode: str = "mci",
    mc_mode: str = "aobmc",
) -> Path:
    """Create a slow-motion version of a video using frame interpolation.

    Args:
        input_path: Path to the input video file
        output_path: Path for the output video (auto-generated if None)
        slowdown: Slow-motion factor (2 = half speed, 3 = third speed, etc.)
        output_fps: Output frame rate (default: 60fps for smooth playback)
        backend: Interpolation backend ('ffmpeg' or 'rife')
        mi_mode: FFmpeg interpolation mode (ffmpeg backend only)
        mc_mode: FFmpeg motion compensation mode (ffmpeg backend only)

    Returns:
        Path to the created slow-motion video
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        suffix = input_path.suffix or ".mp4"
        tag = "rife" if backend == "rife" else "slowmo"
        output_path = input_path.with_name(
            f"{input_path.stem}_{tag}_{slowdown}x{suffix}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Frame Interpolation Settings:")
    print(f"  Input: {input_path}")
    print(f"  Slowdown: {slowdown}x")
    print(f"  Output: {output_path}")

    if backend == "rife":
        interpolate_rife(input_path, output_path, slowdown, output_fps)
    elif backend == "ffmpeg":
        interpolate_ffmpeg(input_path, output_path, slowdown, output_fps, mi_mode, mc_mode)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'ffmpeg' or 'rife'.")

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
  python -m src.interpolate "clip.mp4" --slowdown 3 --backend rife
  python -m src.interpolate "step_05.mp4" --slowdown 4 --fps 60
  python -m src.interpolate "clip.mp4" --slowdown 3 -o "slowmo_clip.mp4"

Backends:
  ffmpeg  - FFmpeg minterpolate (basic quality, no setup needed)
  rife    - RIFE v4.25 on Modal cloud GPU (high quality)
            Requires: pip install modal && modal setup
            First run: python -m modal deploy src/rife_modal.py

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
        "--backend", "-b",
        choices=["ffmpeg", "rife"],
        default="ffmpeg",
        help="Interpolation backend (default: ffmpeg)",
    )

    parser.add_argument(
        "--mi-mode",
        choices=["dup", "blend", "mci"],
        default="mci",
        help="FFmpeg interpolation mode (ffmpeg backend only)",
    )

    parser.add_argument(
        "--mc-mode",
        choices=["obmc", "aobmc"],
        default="aobmc",
        help="FFmpeg motion compensation (ffmpeg backend only)",
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
            backend=args.backend,
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
