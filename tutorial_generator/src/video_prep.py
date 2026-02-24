"""Video preprocessing for cost reduction before Gemini analysis."""

import subprocess
from pathlib import Path
from typing import Optional

import ffmpeg


def get_video_duration(video_path: Path) -> float:
    """Get the duration of a video in seconds.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds
    """
    probe = ffmpeg.probe(str(video_path))
    duration = float(probe['format']['duration'])
    return duration


def preprocess_video(
    input_path: Path,
    output_path: Optional[Path] = None,
    downscale: bool = True,
    downscale_height: int = 480,
    reduce_fps: bool = True,
    target_fps: int = 15,
    trim_intro: int = 0,
    trim_outro: int = 0,
) -> Path:
    """Preprocess video for Gemini API upload (cost reduction).

    Args:
        input_path: Path to the input video
        output_path: Path for the output video (default: input_preprocessed.mp4)
        downscale: Whether to downscale the video
        downscale_height: Target height in pixels (maintains aspect ratio)
        reduce_fps: Whether to reduce frame rate
        target_fps: Target frame rate
        trim_intro: Seconds to trim from the start
        trim_outro: Seconds to trim from the end

    Returns:
        Path to the preprocessed video
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_preprocessed.mp4"
    else:
        output_path = Path(output_path)

    # Get video duration for trim calculations
    duration = get_video_duration(input_path)

    # Calculate trim times
    start_time = trim_intro
    end_time = duration - trim_outro

    if end_time <= start_time:
        raise ValueError(
            f"Invalid trim values: intro={trim_intro}s, outro={trim_outro}s "
            f"exceeds video duration of {duration:.1f}s"
        )

    # Build ffmpeg filter chain
    stream = ffmpeg.input(str(input_path))

    # Apply trimming if needed
    if trim_intro > 0 or trim_outro > 0:
        stream = stream.filter('trim', start=start_time, end=end_time)
        stream = stream.filter('setpts', 'PTS-STARTPTS')

    filters = []

    # Build video filters
    video_stream = stream.video

    if trim_intro > 0 or trim_outro > 0:
        video_stream = video_stream.filter('trim', start=start_time, end=end_time)
        video_stream = video_stream.filter('setpts', 'PTS-STARTPTS')

    if downscale:
        # Scale to target height, maintaining aspect ratio (-2 ensures even width)
        video_stream = video_stream.filter('scale', -2, downscale_height)

    if reduce_fps:
        video_stream = video_stream.filter('fps', fps=target_fps)

    # Handle audio stream
    audio_stream = stream.audio

    if trim_intro > 0 or trim_outro > 0:
        audio_stream = audio_stream.filter('atrim', start=start_time, end=end_time)
        audio_stream = audio_stream.filter('asetpts', 'PTS-STARTPTS')

    # Combine and output
    output = ffmpeg.output(
        video_stream,
        audio_stream,
        str(output_path),
        vcodec='libx264',
        acodec='aac',
        preset='fast',
        crf=23,  # Good quality/size balance
    )

    # Run ffmpeg
    print(f"Preprocessing video...")
    print(f"  - Downscale to {downscale_height}p: {downscale}")
    print(f"  - Reduce FPS to {target_fps}: {reduce_fps}")
    print(f"  - Trim intro: {trim_intro}s")
    print(f"  - Trim outro: {trim_outro}s")

    try:
        output.overwrite_output().run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise

    print(f"Preprocessed video saved to: {output_path}")

    # Show size comparison
    original_size = input_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - new_size / original_size) * 100

    print(f"Size reduction: {original_size:.1f}MB -> {new_size:.1f}MB ({reduction:.1f}% smaller)")

    return output_path


if __name__ == "__main__":
    # Test the preprocessor
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tutorial_generator.src.video_prep <video_path> [trim_intro] [trim_outro]")
        sys.exit(1)

    input_video = Path(sys.argv[1])
    trim_intro = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    trim_outro = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    output = preprocess_video(
        input_video,
        trim_intro=trim_intro,
        trim_outro=trim_outro,
    )
    print(f"Output: {output}")
