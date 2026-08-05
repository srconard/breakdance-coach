"""Create dance-part boundaries with Gemini or a deterministic offline split."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

from tutorial_generator.config import get_api_key


DEFAULT_MODEL = "gemini-2.5-flash"


def segment_video(
    video_path: str | Path,
    duration_s: float,
    fallback_parts: int = 4,
    no_ai: bool = False,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Segment a video, falling back to uniform parts when AI is unavailable."""
    if fallback_parts not in range(3, 9):
        raise ValueError("fallback_parts must be between 3 and 8.")
    if no_ai:
        return uniform_parts(duration_s, fallback_parts)

    try:
        api_key = get_api_key("google")
    except ValueError:
        warnings.warn(
            "No Google API key found; using a uniform offline split. "
            "Set GOOGLE_API_KEY (or GEMINI_API_KEY) to enable Gemini segmentation.",
            stacklevel=2,
        )
        return uniform_parts(duration_s, fallback_parts)

    try:
        raw_parts = analyze_with_gemini(Path(video_path), api_key, model_name)
        repaired = repair_parts(raw_parts, duration_s)
        if len(repaired) < 3:
            warnings.warn(
                "Gemini returned fewer than 3 usable parts; using a uniform split.",
                stacklevel=2,
            )
            return uniform_parts(duration_s, fallback_parts)
        return repaired
    except Exception as exc:
        warnings.warn(
            f"Gemini segmentation failed ({exc}); using a uniform offline split.",
            stacklevel=2,
        )
        return uniform_parts(duration_s, fallback_parts)


def uniform_parts(duration_s: float, count: int = 4) -> list[dict[str, Any]]:
    """Split the full video duration into equal, contiguous practice parts."""
    if duration_s <= 0:
        raise ValueError("duration_s must be greater than zero.")
    if count not in range(3, 9):
        raise ValueError("count must be between 3 and 8.")

    parts = []
    for index in range(count):
        start = duration_s * index / count
        end = duration_s * (index + 1) / count
        parts.append(
            {
                "title": f"Part {index + 1}",
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "counts": None,
                "cue": "",
            }
        )
    return parts


def analyze_with_gemini(
    video_path: Path,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Upload a clip to Gemini and request a bare array of dance parts."""
    from google import genai
    from google.genai import types

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    client = genai.Client(api_key=api_key)
    uploaded = None
    try:
        print(f"Uploading video to Gemini: {video_path.name}")
        uploaded = client.files.upload(file=str(video_path))
        print("Waiting for Gemini to process video", end="", flush=True)
        while not any(word in str(uploaded.state).upper() for word in ("ACTIVE", "FAILED")):
            print(".", end="", flush=True)
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        print(" Done")
        if "FAILED" in str(uploaded.state).upper():
            raise RuntimeError("Gemini video processing failed.")

        prompt = """Analyze this short dance video as practice material. Return 3 to 8 PARTS that cover the whole danced section. Parts are larger learnable phrases, not individual tutorial steps.

For each part return exactly:
{"start_s": float, "end_s": float, "title": "2-4 words", "cue": "one plain-language sentence"}

Rules:
- Parts must be in chronological order, contiguous, and non-overlapping.
- Use precise timestamps in seconds.
- Describe only visible body movement in each cue, for example: "Step right as the left arm sweeps across on count 3."
- Never invent or assert a technical move name unless it is unambiguous from the video; prefer literal body-action language.
- Exclude captions, creator identity, music interpretation, and commentary.
- Return ONLY a bare valid JSON array. No markdown fences or explanation."""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "start_s": types.Schema(type=types.Type.NUMBER),
                        "end_s": types.Schema(type=types.Type.NUMBER),
                        "title": types.Schema(type=types.Type.STRING),
                        "cue": types.Schema(type=types.Type.STRING),
                    },
                    required=["start_s", "end_s", "title", "cue"],
                ),
            ),
        )
        for attempt in range(2):
            response = client.models.generate_content(
                model=model_name,
                contents=[uploaded, prompt],
                config=config,
            )
            try:
                response_text = _strip_json_fence(response.text)
                data = json.loads(response_text)
                if not isinstance(data, list):
                    raise ValueError("Gemini response was not a JSON array.")
                return data
            except (json.JSONDecodeError, ValueError):
                if attempt == 1:
                    raise
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def _strip_json_fence(text: str) -> str:
    """Remove optional Markdown JSON fences from a model response."""
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        value = "\n".join(lines).strip()
    return value


def repair_parts(raw_parts: list[dict[str, Any]], duration_s: float) -> list[dict[str, Any]]:
    """Clamp, order, join, and if necessary merge model-produced parts."""
    cleaned = []
    for index, raw in enumerate(raw_parts):
        if not isinstance(raw, dict):
            continue
        try:
            start = max(0.0, min(duration_s, float(raw["start_s"])))
            end = max(0.0, min(duration_s, float(raw["end_s"])))
        except (KeyError, TypeError, ValueError):
            continue
        if end - start <= 0.001:
            continue
        title = str(raw.get("title") or f"Part {index + 1}").strip()
        cue = str(raw.get("cue") or "").strip()
        cleaned.append(
            {
                "title": title,
                "start_s": start,
                "end_s": end,
                "counts": None,
                "cue": cue,
            }
        )

    cleaned.sort(key=lambda part: (part["start_s"], part["end_s"]))
    if not cleaned:
        return []

    # Convert suggested transitions into one strictly increasing boundary list.
    # This also repairs nested or heavily overlapping model ranges.
    section_start = cleaned[0]["start_s"]
    section_end = max(part["end_s"] for part in cleaned)
    section_duration = section_end - section_start
    if section_duration < 0.003 * len(cleaned):
        return []
    minimum_span = min(0.01, section_duration / (len(cleaned) * 2))
    boundaries = [section_start]
    for index in range(len(cleaned) - 1):
        suggested = (cleaned[index]["end_s"] + cleaned[index + 1]["start_s"]) / 2
        lower = boundaries[-1] + minimum_span
        remaining_parts = len(cleaned) - index - 1
        upper = section_end - minimum_span * remaining_parts
        boundaries.append(max(lower, min(upper, suggested)))
    boundaries.append(section_end)
    for index, part in enumerate(cleaned):
        part["start_s"] = boundaries[index]
        part["end_s"] = boundaries[index + 1]
    while len(cleaned) > 8:
        pair_index = min(
            range(len(cleaned) - 1),
            key=lambda i: cleaned[i + 1]["end_s"] - cleaned[i]["start_s"],
        )
        first, second = cleaned[pair_index : pair_index + 2]
        merged = {
            "title": first["title"],
            "start_s": first["start_s"],
            "end_s": second["end_s"],
            "counts": None,
            "cue": " ".join(value for value in (first["cue"], second["cue"]) if value),
        }
        cleaned[pair_index : pair_index + 2] = [merged]

    for part in cleaned:
        part["start_s"] = round(part["start_s"], 3)
        part["end_s"] = round(part["end_s"], 3)
    return cleaned
