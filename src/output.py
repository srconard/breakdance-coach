"""Output generator for creating Obsidian-compatible Markdown files."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from src.video_analyzer import TutorialStep


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


def generate_markdown(
    title: str,
    steps: list[TutorialStep],
    descriptions: list[str],
    gif_paths: list[Path],
    output_dir: Path,
    source_url: Optional[str] = None,
    template_dir: Optional[Path] = None,
) -> Path:
    """Generate an Obsidian-compatible Markdown file with embedded GIFs.

    Args:
        title: Title of the tutorial
        steps: List of tutorial steps
        descriptions: List of descriptions for each step
        gif_paths: List of paths to GIF files
        output_dir: Directory for output files
        source_url: Optional YouTube URL for reference
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
    for step, desc, gif_path in zip(steps, descriptions, gif_paths):
        # Copy GIF to output directory
        new_gif_path = gifs_dir / gif_path.name
        if gif_path != new_gif_path:
            shutil.copy2(gif_path, new_gif_path)

        step_data = StepWithDescription(
            step_number=step.step_number,
            start_time=step.start_time,
            end_time=step.end_time,
            label=step.label,
            description=desc,
            gif_path=new_gif_path,
            gif_filename=f"gifs/{gif_path.name}",  # Relative path for Obsidian
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
    from src.video_analyzer import TutorialStep

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
