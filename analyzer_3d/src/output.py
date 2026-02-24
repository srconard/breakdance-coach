"""Obsidian markdown output for 3D move analysis."""

import json
import shutil
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader


def generate_3d_markdown(
    title: str,
    glb_paths: list[Path],
    output_dir: Path,
    backend: str = "gvhmr",
    source_url: Optional[str] = None,
    step_labels: Optional[list[str]] = None,
    template_dir: Optional[Path] = None,
) -> Path:
    """Generate Obsidian-compatible Markdown with embedded 3D models.

    Args:
        title: Title of the move/tutorial
        glb_paths: List of GLB file paths (one per step or one for full video)
        output_dir: Directory for output files
        backend: Pose estimation backend used ('gvhmr' or 'deepmotion')
        source_url: YouTube URL (if applicable)
        step_labels: Optional labels for each step
        template_dir: Directory containing Jinja2 templates

    Returns:
        Path to the generated Markdown file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create models subdirectory
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Copy GLB files to output
    model_filenames = []
    for glb_path in glb_paths:
        dest = models_dir / glb_path.name
        if glb_path != dest:
            shutil.copy2(glb_path, dest)
        model_filenames.append(f"models/{glb_path.name}")

    # Build step data
    steps = []
    for i, (glb_file, glb_path) in enumerate(zip(model_filenames, glb_paths)):
        label = step_labels[i] if step_labels and i < len(step_labels) else f"Clip {i + 1}"
        steps.append({
            "number": i + 1,
            "label": label,
            "model_filename": glb_file,
        })

    # Load and render template
    if template_dir is None:
        template_dir = Path(__file__).parent.parent / "templates"

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("3d_tutorial.md")

    from datetime import date
    content = template.render(
        title=title,
        source_url=source_url,
        backend=backend,
        generated_date=date.today().isoformat(),
        steps=steps,
    )

    # Sanitize title for filename
    import re
    safe_title = re.sub(r'[<>:"/\\|?*]+', '_', title)
    safe_title = re.sub(r'[\s_]+', '_', safe_title).strip('_')[:100]

    output_file = output_dir / f"{safe_title}_3D.md"
    output_file.write_text(content, encoding="utf-8")

    # Save metadata
    metadata = {
        "title": title,
        "source_url": source_url,
        "backend": backend,
        "models": model_filenames,
        "steps": steps,
    }
    metadata_path = output_dir / "3d_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Generated 3D Markdown: {output_file}")
    print(f"Models copied to: {models_dir}")

    return output_file
