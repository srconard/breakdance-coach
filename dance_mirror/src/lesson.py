"""Load, validate, and save the editable Dance Mirror lesson schema."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def validate_lesson(lesson: dict[str, Any]) -> dict[str, Any]:
    """Validate a lesson dictionary and return it unchanged.

    The validator is intentionally strict about timing and types so a hand-edited
    JSON file fails with a useful path instead of producing a broken practice page.
    """
    if not isinstance(lesson, dict):
        raise ValueError("Lesson must be a JSON object.")

    required = {
        "schema_version",
        "title",
        "source_url",
        "creator_credit",
        "video_file",
        "duration_s",
        "fps",
        "parts",
    }
    missing = sorted(required - lesson.keys())
    if missing:
        raise ValueError(f"Lesson is missing required field(s): {', '.join(missing)}")
    if lesson["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {lesson['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}."
        )

    for key in ("title", "video_file"):
        if not isinstance(lesson[key], str) or not lesson[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    for key in ("source_url", "creator_credit"):
        if lesson[key] is not None and not isinstance(lesson[key], str):
            raise ValueError(f"{key} must be a string or null.")

    duration = _finite_number(lesson["duration_s"], "duration_s")
    fps = _finite_number(lesson["fps"], "fps")
    if duration <= 0:
        raise ValueError("duration_s must be greater than zero.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    parts = lesson["parts"]
    if not isinstance(parts, list) or not 3 <= len(parts) <= 8:
        raise ValueError("parts must contain between 3 and 8 entries.")

    for index, part in enumerate(parts):
        path = f"parts[{index}]"
        if not isinstance(part, dict):
            raise ValueError(f"{path} must be an object.")
        for key in ("title", "start_s", "end_s", "cue"):
            if key not in part:
                raise ValueError(f"{path} is missing required field {key!r}.")
        if not isinstance(part["title"], str) or not part["title"].strip():
            raise ValueError(f"{path}.title must be a non-empty string.")
        if not isinstance(part["cue"], str):
            raise ValueError(f"{path}.cue must be a string.")
        if "counts" in part and part["counts"] is not None and not isinstance(part["counts"], str):
            raise ValueError(f"{path}.counts must be a string or null.")

        start = _finite_number(part["start_s"], f"{path}.start_s")
        end = _finite_number(part["end_s"], f"{path}.end_s")
        if start < 0 or end > duration + 0.001:
            raise ValueError(f"{path} must stay within 0 and duration_s ({duration:.3f}).")
        if end <= start:
            raise ValueError(f"{path}.end_s must be greater than start_s.")
    return lesson


def _finite_number(value: Any, path: str) -> float:
    """Return a finite float or raise a schema-oriented error."""
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number.")
    return result


def load_lesson(path: str | Path) -> dict[str, Any]:
    """Load and validate a Dance Mirror JSON file."""
    lesson_path = Path(path)
    try:
        data = json.loads(lesson_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {lesson_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    return validate_lesson(data)


def save_lesson(lesson: dict[str, Any], path: str | Path) -> Path:
    """Validate and save a Dance Mirror lesson as readable UTF-8 JSON."""
    validate_lesson(lesson)
    lesson_path = Path(path)
    lesson_path.parent.mkdir(parents=True, exist_ok=True)
    lesson_path.write_text(
        json.dumps(lesson, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return lesson_path
