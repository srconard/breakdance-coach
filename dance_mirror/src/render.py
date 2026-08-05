"""Render a self-contained, offline Dance Mirror practice page."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from dance_mirror.src.lesson import validate_lesson


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PACKAGE_DIR / "templates" / "lesson.html"


def render_lesson(
    lesson: dict[str, Any],
    lesson_dir: str | Path,
    output_path: str | Path | None = None,
    template_path: str | Path = DEFAULT_TEMPLATE,
) -> Path:
    """Inline lesson JSON and its source video into a standalone HTML file."""
    validate_lesson(lesson)
    lesson_directory = Path(lesson_dir)
    video_path = lesson_directory / lesson["video_file"]
    if not video_path.is_file():
        raise FileNotFoundError(f"Lesson video not found: {video_path}")

    mime_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    video_data_uri = f"data:{mime_type};base64,{video_b64}"
    lesson_json = json.dumps(lesson, ensure_ascii=False).replace("</", "<\\/")

    template = Path(template_path).read_text(encoding="utf-8")
    html = template.replace("{{LESSON_JSON}}", lesson_json)
    html = html.replace("{{VIDEO_DATA_URI}}", video_data_uri)
    if "{{LESSON_JSON}}" in html or "{{VIDEO_DATA_URI}}" in html:
        raise ValueError("Lesson template contains unresolved placeholders.")

    destination = Path(output_path) if output_path else lesson_directory / "lesson.html"
    destination.write_text(html, encoding="utf-8")
    return destination

