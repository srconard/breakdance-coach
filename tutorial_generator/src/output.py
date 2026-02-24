"""Output generator for creating Obsidian-compatible Markdown files and metadata."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from tutorial_generator.src.video_analyzer import TutorialStep


@dataclass
class StepWithDescription:
    """A tutorial step with its description and GIF path."""

    step_number: int
    start_time: str
    end_time: str
    label: str
    description: str
    gif_path: Path
    gif_filename: str


def save_metadata(
    title: str,
    steps: list[TutorialStep],
    descriptions: list[str],
    clip_filenames: list[str],
    output_dir: Path,
    source_url: Optional[str] = None,
    original_video: Optional[str] = None,
    clip_settings: Optional[dict] = None,
) -> Path:
    """Save tutorial metadata as JSON for later re-clipping and reference.

    Args:
        title: Title of the tutorial
        steps: List of tutorial steps with timestamps
        descriptions: List of descriptions for each step
        clip_filenames: List of clip filenames (relative to gifs/)
        output_dir: Directory for output files
        source_url: YouTube URL (if downloaded)
        original_video: Path to the original full-quality video
        clip_settings: Settings used to create clips (fps, width, format)

    Returns:
        Path to the saved JSON file
    """
    output_dir = Path(output_dir)

    metadata = {
        "title": title,
        "source_url": source_url,
        "original_video": str(original_video) if original_video else None,
        "clip_settings": clip_settings or {},
        "steps": [],
    }

    for step, desc, clip_file in zip(steps, descriptions, clip_filenames):
        metadata["steps"].append({
            "step_number": step.step_number,
            "start_time": step.start_time,
            "end_time": step.end_time,
            "start_seconds": step.start_seconds,
            "end_seconds": step.end_seconds,
            "label": step.label,
            "description": desc,
            "clip_filename": clip_file,
        })

    metadata_path = output_dir / "tutorial_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved metadata: {metadata_path}")
    return metadata_path


def generate_markdown(
    title: str,
    steps: list[TutorialStep],
    descriptions: list[str],
    gif_paths: list[Path],
    output_dir: Path,
    source_url: Optional[str] = None,
    original_video: Optional[Path] = None,
    clip_settings: Optional[dict] = None,
    template_dir: Optional[Path] = None,
) -> Path:
    """Generate an Obsidian-compatible Markdown file with embedded GIFs.

    Also saves a tutorial_metadata.json alongside the markdown for
    re-clipping at different quality levels later.

    Args:
        title: Title of the tutorial
        steps: List of tutorial steps
        descriptions: List of descriptions for each step
        gif_paths: List of paths to GIF files
        output_dir: Directory for output files
        source_url: Optional YouTube URL for reference
        original_video: Path to the original full-quality video
        clip_settings: Settings used to create clips (fps, width, format)
        template_dir: Directory containing Jinja2 templates

    Returns:
        Path to the generated Markdown file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create gifs subdirectory
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(exist_ok=True)

    # Prepare step data with descriptions
    steps_with_desc = []
    clip_filenames = []
    for step, desc, gif_path in zip(steps, descriptions, gif_paths):
        # Copy GIF to output directory
        new_gif_path = gifs_dir / gif_path.name
        if gif_path != new_gif_path:
            shutil.copy2(gif_path, new_gif_path)

        relative_filename = f"gifs/{gif_path.name}"
        clip_filenames.append(relative_filename)

        step_data = StepWithDescription(
            step_number=step.step_number,
            start_time=step.start_time,
            end_time=step.end_time,
            label=step.label,
            description=desc,
            gif_path=new_gif_path,
            gif_filename=relative_filename,
        )
        steps_with_desc.append(step_data)

    # Load template
    if template_dir is None:
        template_dir = Path(__file__).parent.parent / "templates"

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("tutorial.md")

    # Render template
    content = template.render(
        title=title,
        source_url=source_url,
        steps=steps_with_desc,
    )

    # Write output file
    output_file = output_dir / f"{sanitize_title(title)}.md"
    output_file.write_text(content, encoding="utf-8")

    # Save metadata JSON for re-clipping
    save_metadata(
        title=title,
        steps=steps,
        descriptions=descriptions,
        clip_filenames=clip_filenames,
        output_dir=output_dir,
        source_url=source_url,
        original_video=original_video,
        clip_settings=clip_settings,
    )

    print(f"\nGenerated Markdown: {output_file}")
    print(f"GIFs copied to: {gifs_dir}")

    return output_file


def sanitize_title(title: str) -> str:
    """Sanitize a title for use as a filename.

    Args:
        title: The title to sanitize

    Returns:
        A filename-safe string
    """
    import re

    # Replace special chars with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]+', '_', title)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rsplit('_', 1)[0]

    return sanitized


if __name__ == "__main__":
    # Test the output generator
    from tutorial_generator.src.video_analyzer import TutorialStep

    test_steps = [
        TutorialStep(1, "00:15", "00:45", "Basic Stance"),
        TutorialStep(2, "00:45", "01:30", "First Move"),
    ]

    test_descriptions = [
        "Stand with your feet shoulder-width apart. Keep your weight centered and your knees slightly bent for mobility.",
        "Step to the right with your right foot, then bring your left foot to meet it. Repeat this side-to-side motion.",
    ]

    # This would require actual GIF files to test fully
    print("Output generator module loaded successfully.")
    print(f"Sanitized title test: '{sanitize_title('Test: Tutorial / Video')}'")
