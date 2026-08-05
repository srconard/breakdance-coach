"""Build and refresh self-contained Dance Mirror practice lessons."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from dance_mirror.src.lesson import load_lesson, save_lesson
from dance_mirror.src.render import render_lesson
from dance_mirror.src.segmenter import segment_video
from shared.downloader import download_video
from tutorial_generator.src.video_prep import get_video_duration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Dance Mirror build and refresh subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="ingest a video and build a lesson")
    inputs = build.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--local-file", metavar="PATH", help="local dance clip")
    inputs.add_argument("--url", help="TikTok, YouTube, or Instagram video URL")
    build.add_argument("--title", help="lesson title (defaults to the clip title)")
    build.add_argument("--credit", default=None, help="creator credit, such as @dancer")
    build.add_argument("--source-url", default=None, help="canonical source link")
    build.add_argument(
        "--parts",
        type=int,
        choices=range(3, 9),
        default=4,
        metavar="N",
        help="offline fallback part count, 3-8 (default: 4)",
    )
    build.add_argument("--no-ai", action="store_true", help="skip Gemini and split uniformly")
    build.add_argument(
        "-o",
        "--output-dir",
        help="lesson directory (default: output/dance_mirror/<slug>)",
    )

    refresh = subparsers.add_parser(
        "refresh", help="re-render lesson.html from an edited dance_lesson.json"
    )
    refresh.add_argument("lesson_dir", help="directory containing dance_lesson.json")
    return parser.parse_args(argv)


def slugify(value: str) -> str:
    """Create a filesystem-friendly lowercase lesson slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dance-lesson"


def get_video_fps(video_path: Path) -> float:
    """Read average video frame rate with ffprobe."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            raise ValueError("no video stream found")
        rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
        fps = float(Fraction(rate))
        if fps <= 0:
            raise ValueError("frame rate is zero")
        return fps
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, ZeroDivisionError) as exc:
        raise RuntimeError(f"Could not read video frame rate with ffprobe: {exc}") from exc


def build_lesson(args: argparse.Namespace) -> Path:
    """Ingest, segment, save, and render a Dance Mirror lesson."""
    temp_dir: Path | None = None
    if args.local_file:
        input_video = Path(args.local_file)
        if not input_video.is_file():
            raise FileNotFoundError(f"Local video not found: {input_video}")
        title = args.title or input_video.stem.replace("_", " ").replace("-", " ").strip()
        source_url = args.source_url
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="dance_mirror_"))
        try:
            input_video, downloaded_title = download_video(
                args.url,
                output_dir=str(temp_dir),
                filename="source",
                quality="best_mp4",
            )
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                "Could not download that video. It may be private, blocked, or unavailable "
                "from this network. Save the video yourself, then run Dance Mirror with "
                "--local-file. Dance Mirror does not bypass access controls."
            ) from exc
        title = args.title or downloaded_title
        source_url = args.source_url or args.url

    lesson_dir = Path(args.output_dir) if args.output_dir else Path("output/dance_mirror") / slugify(title)
    lesson_dir.mkdir(parents=True, exist_ok=True)
    source_video = lesson_dir / "source.mp4"

    try:
        if input_video.resolve() != source_video.resolve():
            shutil.copy2(input_video, source_video)
        duration_s = get_video_duration(source_video)
        fps = get_video_fps(source_video)
        parts = segment_video(
            source_video,
            duration_s=duration_s,
            fallback_parts=args.parts,
            no_ai=args.no_ai,
        )
        lesson = {
            "schema_version": 1,
            "title": title,
            "source_url": source_url,
            "creator_credit": args.credit,
            "video_file": "source.mp4",
            "duration_s": round(duration_s, 3),
            "fps": round(fps, 6),
            "parts": parts,
        }
        json_path = save_lesson(lesson, lesson_dir / "dance_lesson.json")
        html_path = render_lesson(lesson, lesson_dir)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Lesson JSON: {json_path}")
    print(f"Practice page: {html_path}")
    print("Open lesson.html directly in a browser to practice offline.")
    return lesson_dir


def refresh_lesson(lesson_dir: str | Path) -> Path:
    """Render HTML from an existing lesson JSON without invoking AI."""
    directory = Path(lesson_dir)
    json_path = directory / "dance_lesson.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Lesson JSON not found: {json_path}")
    lesson = load_lesson(json_path)
    output = render_lesson(lesson, directory)
    print(f"Refreshed practice page: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    """Run the selected Dance Mirror command."""
    args = parse_args(argv)
    try:
        if args.command == "build":
            build_lesson(args)
        else:
            refresh_lesson(args.lesson_dir)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
