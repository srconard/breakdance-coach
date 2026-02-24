"""CLI interface for the Breakdance Tutorial GIF Generator."""

import argparse
import sys
import tempfile
from pathlib import Path

from tutorial_generator.config import Settings
from shared.downloader import download_video
from tutorial_generator.src.video_prep import preprocess_video
from tutorial_generator.src.video_analyzer import analyze_video, print_steps
from tutorial_generator.src.description import get_description_provider
from tutorial_generator.src.gif_creator import create_clips_for_steps
from tutorial_generator.src.output import generate_markdown


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate GIF tutorials from YouTube breakdancing videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tutorial_generator.src.main --local-file "video.mp4" --title "My Tutorial"
  python -m tutorial_generator.src.main --local-file "video.mp4" --title "Tutorial" --format mp4
  python -m tutorial_generator.src.main --local-file "video.mp4" --title "Tutorial" --trim-intro 10
  python -m tutorial_generator.src.main "https://youtube.com/watch?v=..." --description-model anthropic
        """,
    )

    parser.add_argument(
        "url",
        nargs="?",  # Make URL optional when using --local-file
        help="YouTube video URL (not required if using --local-file)",
    )

    parser.add_argument(
        "--local-file",
        type=str,
        metavar="PATH",
        help="Use a local video file instead of downloading from YouTube",
    )

    parser.add_argument(
        "--source-url",
        type=str,
        metavar="URL",
        help="YouTube URL to store in metadata (useful with --local-file)",
    )

    parser.add_argument(
        "--title",
        type=str,
        help="Video title (required when using --local-file)",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output folder name (default: video title)",
        default=None,
    )

    parser.add_argument(
        "--description-model",
        choices=["google", "anthropic", "openai"],
        default="google",
        help="LLM provider for generating descriptions (default: google)",
    )

    # Cost reduction options
    parser.add_argument(
        "--no-downscale",
        action="store_true",
        help="Keep original video resolution (disables 480p downscale)",
    )

    parser.add_argument(
        "--no-fps-reduce",
        action="store_true",
        help="Keep original frame rate (disables 15fps reduction)",
    )

    parser.add_argument(
        "--trim-intro",
        type=int,
        default=0,
        metavar="N",
        help="Trim N seconds from the start of the video",
    )

    parser.add_argument(
        "--trim-outro",
        type=int,
        default=0,
        metavar="N",
        help="Trim N seconds from the end of the video",
    )

    # Output format options
    parser.add_argument(
        "--format",
        choices=["gif", "mp4", "webm"],
        default="gif",
        help="Output format: gif (default), mp4 (with playback controls), or webm (smallest)",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        metavar="N",
        help="Output frame rate (default: 12)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        metavar="N",
        help="Output width in pixels (default: 640)",
    )

    # Advanced options
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (downloaded/preprocessed video)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()

    # Validate arguments
    if not args.local_file and not args.url:
        print("Error: Either provide a YouTube URL or use --local-file")
        return 1

    if args.local_file and not args.title:
        print("Error: --title is required when using --local-file")
        return 1

    print("=" * 60)
    print("BREAKDANCE TUTORIAL GIF GENERATOR")
    print("=" * 60)

    # Create temp directory for intermediate files
    temp_dir = Path(tempfile.mkdtemp(prefix="breakdance_"))
    print(f"\nTemp directory: {temp_dir}")

    try:
        # Step 1: Get video (download or use local file)
        if args.local_file:
            print("\n[1/6] Using local video file...")
            video_path = Path(args.local_file)
            if not video_path.exists():
                raise FileNotFoundError(f"Local file not found: {video_path}")
            video_title = args.title
            source_url = args.source_url  # Can be set via --source-url
            print(f"Video: {video_path}")
        else:
            print("\n[1/6] Downloading video...")
            video_path, video_title = download_video(
                args.url,
                output_dir=str(temp_dir),
            )
            source_url = args.source_url or args.url

        # Determine output directory
        if args.output:
            output_dir = Path("output") / args.output
        else:
            from tutorial_generator.src.output import sanitize_title
            output_dir = Path("output") / sanitize_title(video_title)

        # Step 2: Preprocess video (cost reduction)
        print("\n[2/6] Preprocessing video for analysis...")
        preprocessed_path = preprocess_video(
            video_path,
            downscale=not args.no_downscale,
            reduce_fps=not args.no_fps_reduce,
            trim_intro=args.trim_intro,
            trim_outro=args.trim_outro,
        )

        # Step 3: Analyze video with Gemini
        print("\n[3/6] Analyzing video with Gemini...")
        steps = analyze_video(preprocessed_path)
        print_steps(steps)

        if not steps:
            print("\nNo tutorial steps found. Exiting.")
            return 1

        # Step 4: Generate descriptions
        print(f"\n[4/6] Generating descriptions with {args.description_model}...")
        provider = get_description_provider(args.description_model)
        descriptions = provider.generate_descriptions(steps, video_title)

        # Step 5: Create clips (GIF or video)
        # Use the original video for better quality
        format_label = "GIFs" if args.format == "gif" else f"{args.format.upper()} videos"
        print(f"\n[5/6] Creating {format_label} from original video...")

        # If video was trimmed, we need to adjust timestamps since
        # analysis was done on preprocessed video but clips use original
        clip_steps = steps
        if args.trim_intro > 0:
            # Create adjusted steps with offset timestamps
            from tutorial_generator.src.video_analyzer import TutorialStep
            clip_steps = []
            for step in steps:
                adjusted_step = TutorialStep(
                    step_number=step.step_number,
                    start_time=step.start_time,  # Will be recalculated
                    end_time=step.end_time,      # Will be recalculated
                    label=step.label,
                )
                # Override the calculated seconds with adjusted values
                adjusted_step.start_seconds = step.start_seconds + args.trim_intro
                adjusted_step.end_seconds = step.end_seconds + args.trim_intro
                clip_steps.append(adjusted_step)

        clip_paths = create_clips_for_steps(
            video_path=video_path,  # Use original video for better quality
            steps=clip_steps,
            output_dir=temp_dir / "clips",
            fps=args.fps,
            width=args.width,
            format=args.format,
        )

        # Step 6: Generate Markdown output
        print("\n[6/6] Generating Markdown output...")
        output_file = generate_markdown(
            title=video_title,
            steps=steps,
            descriptions=descriptions,
            gif_paths=clip_paths,
            output_dir=output_dir,
            source_url=source_url,
            original_video=video_path,
            clip_settings={
                "fps": args.fps,
                "width": args.width,
                "format": args.format,
            },
        )

        print("\n" + "=" * 60)
        print("COMPLETE!")
        print("=" * 60)
        print(f"\nOutput directory: {output_dir}")
        print(f"Markdown file: {output_file}")
        print(f"{format_label}: {output_dir / 'gifs'}")
        print("\nOpen the Markdown file in Obsidian to view your tutorial!")

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 130

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup temp files
        if not args.keep_temp and temp_dir.exists():
            import shutil
            print(f"\nCleaning up temp files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
