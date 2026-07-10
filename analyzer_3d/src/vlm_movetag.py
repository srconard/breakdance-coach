"""VLM move-tagger for breaking videos.

Gemini watches a clip and segments it into named breaking moves from a
CONTROLLED vocabulary, so freestyle SEQUENCES (AIST++/FineDance) can be
indexed by the by-move reference library (move_library.py).

Output contract (strict JSON):
    {"video": ..., "model": ..., "segments":
        [{"move", "start_s", "end_s", "confidence", "notes"}]}
Segments are non-overlapping and sorted; gaps are allowed for idle /
walking stretches. Move names are validated against MOVE_VOCAB — an
out-of-vocabulary label triggers one corrective retry, then is coerced
to "other" with the original label preserved in notes.

Usage:
    python -m analyzer_3d.src.vlm_movetag video.mp4
        [--out tags.json] [--model gemini-2.5-pro] [--fps 5]
        [--merge-into output/reference_library/move_tags.json]

--merge-into upserts the result into a merged JSON keyed by the
LIBRARY STEM (camera slot replaced with "cAll", e.g.
gBR_sBM_c01_d04_mBR0_ch01 -> gBR_sBM_cAll_d04_mBR0_ch01) so tags can
be joined to the aistpp reference-library animations.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment; falls
back to reading GEMINI_API_KEY from ~/.claude/.env.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Same default as vlm_annotate.py: pro over flash for temporal accuracy.
# ~$0.03/clip at 5 fps sampling on ~12 s clips.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SAMPLE_FPS = 5.0
MIN_SEG_SEC = 0.15          # drop segments shorter than this
OVERLAP_DROP_SEC = 0.05     # drop a segment swallowed past this tolerance

# Controlled move vocabulary. Keep in sync with the prompt below.
MOVE_VOCAB = [
    "toprock", "uprock", "6-step", "footwork-other", "cc", "swipe",
    "windmill", "flare", "1990", "2000", "headspin", "backspin", "halo",
    "baby-freeze", "chair-freeze", "airchair", "elbow-freeze",
    "handstand-freeze", "kick-up", "transition", "power-combo", "other",
]


# ---------------------------------------------------------------- setup

def load_api_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env_file = Path.home() / ".claude" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r'^\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*"?([^"\s]+)"?\s*$', line)
            if m:
                return m.group(2)
    raise SystemExit("No GEMINI_API_KEY/GOOGLE_API_KEY in environment or ~/.claude/.env")


def probe_video(path):
    """Return dict(width, height, fps, duration) via ffprobe, or None."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
        info = json.loads(out)
        stream = info["streams"][0]
        num, den = stream["r_frame_rate"].split("/")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(num) / float(den),
            "duration": float(info["format"]["duration"]),
        }
    except Exception as e:  # noqa: BLE001 - probe is best-effort
        print(f"[movetag] ffprobe failed ({e}); proceeding without local probe")
        return None


def library_stem(video_path):
    """Camera-agnostic library stem: gBR_sBM_c01_... -> gBR_sBM_cAll_..."""
    return re.sub(r"_c\d{2}_", "_cAll_", Path(video_path).stem)


# ---------------------------------------------------------------- prompt

def build_prompt(duration, sample_fps, probe):
    dur_txt = f"{duration:.2f} seconds" if duration else "unknown length"
    res_txt = f"{probe['width']}x{probe['height']} @ {probe['fps']:.1f} fps" if probe else "unknown resolution"
    vocab_txt = ", ".join(MOVE_VOCAB)
    return f"""You are a breaking (breakdance) judge annotating a dance video for a move library.
The clip is {dur_txt}, {res_txt}. You see frames sampled at {sample_fps:g} fps, so your timestamps are accurate to roughly +/-{1.0 / sample_fps:.1f}s.

TASK: Segment the clip into the breaking moves the dancer performs, in time order.

CONTROLLED VOCABULARY - the "move" field MUST be exactly one of:
{vocab_txt}

Definitions / disambiguation:
- toprock: upright standing dance steps (Indian step, side steps, cross steps).
- uprock: upright battle-style burns/jerks toward an opponent.
- 6-step: the classic 6-step footwork pattern circling on the floor.
- footwork-other: any other floor footwork (3-step, 2-step, coffee grinder/helicopter, shuffles, kick-outs) that is not a clean 6-step.
- cc: CCs / cross-checks - rocking side-to-side on hands in footwork position.
- swipe: torso-twisting swipes where legs whip over and hands alternate.
- windmill: continuous rolling power move on shoulders/back, legs in wide V.
- flare: legs swing in circles around the body supported on alternating hands (no shoulder roll).
- 1990: spinning one-hand handstand. 2000: spinning two-hand handstand.
- headspin: spinning on the head. backspin: spinning on the upper back. halo: head-assisted rolling spin around the head/shoulders.
- baby-freeze / chair-freeze / airchair / elbow-freeze / handstand-freeze: held static poses; pick the closest named freeze.
- kick-up: kip-up / kick-up from the floor back to standing.
- transition: setups, go-downs (drops), stand-ups, idle standing, walking, small in-between steps that are not a recognizable move.
- power-combo: several power moves blended so tightly they cannot be split cleanly (say which ones in notes).
- other: a recognizable move that is NOT in the vocabulary (name it in notes).

Rules:
- Segments must be in chronological order and must NOT overlap.
- Together they should cover the whole clip from first to last movement; you may leave gaps only for idle/walking stretches (or label them "transition").
- Timestamps in seconds with one decimal place; start_s < end_s.
- "confidence": your confidence 0.0-1.0 that the label is the right vocabulary item.
- "notes": short free text - style details, variation names, or the real move name when using "other"/"power-combo"/"footwork-other".
- Prefer the most specific vocabulary item; do not invent new labels.

Return STRICT JSON only (no markdown, no commentary), exactly this shape:
{{
  "segments": [
    {{"move": "toprock", "start_s": 0.0, "end_s": 3.4, "confidence": 0.9, "notes": "indian step"}}
  ]
}}"""


# ---------------------------------------------------------------- gemini

def tag_video(video_path, model=DEFAULT_MODEL, sample_fps=DEFAULT_SAMPLE_FPS, attempts=2):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    probe = probe_video(video_path)
    duration = probe["duration"] if probe else None

    print(f"[movetag] uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)...")
    f = client.files.upload(file=str(video_path))
    usage_totals = {"prompt_tokens": 0, "output_tokens": 0}
    try:
        t0 = time.time()
        while f.state and f.state.name == "PROCESSING":
            if time.time() - t0 > 300:
                raise RuntimeError("Gemini file processing timed out after 300s")
            time.sleep(2)
            f = client.files.get(name=f.name)
        if not f.state or f.state.name != "ACTIVE":
            raise RuntimeError(f"upload ended in state {f.state.name if f.state else 'unknown'}")
        print(f"[movetag] file ACTIVE after {time.time() - t0:.0f}s; asking {model} (sampling {sample_fps:g} fps)...")

        video_part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type or "video/mp4"),
            video_metadata=types.VideoMetadata(fps=sample_fps),
        )
        prompt = build_prompt(duration, sample_fps, probe)

        segments, coerced = None, []
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[
                        video_part,
                        types.Part(text=prompt),
                    ])],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                usage = resp.usage_metadata
                if usage:
                    usage_totals["prompt_tokens"] += usage.prompt_token_count or 0
                    usage_totals["output_tokens"] += usage.candidates_token_count or 0
                    print(f"[movetag] tokens: {usage.prompt_token_count}+{usage.candidates_token_count}")

                raw = resp.text or ""
                raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
                data = json.loads(raw)
                if "segments" not in data or not isinstance(data["segments"], list):
                    raise ValueError(f"response missing 'segments' list: {list(data.keys())}")

                segments, coerced = clean_segments(data["segments"], duration,
                                                   strict=(attempt < attempts))
                break
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                print(f"[movetag] attempt {attempt}/{attempts} invalid ({e}); "
                      f"{'retrying' if attempt < attempts else 'giving up'}...")
                time.sleep(3)
        if segments is None:
            raise RuntimeError(f"VLM tagging failed after {attempts} attempts: {last_err}")
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass

    covered = sum(s["end_s"] - s["start_s"] for s in segments)
    coverage = covered / duration if duration else None
    if coverage is not None and coverage < 0.7:
        print(f"[movetag] WARNING: segments cover only {coverage * 100:.0f}% of the clip")
    if coerced:
        print(f"[movetag] WARNING: coerced out-of-vocab labels to 'other': {coerced}")

    return {
        "_comment": (f"VLM move tags ({model}, sampled {sample_fps:g} fps). "
                     "Controlled vocabulary; joins to reference_library animations by library_stem."),
        "video": video_path.name,
        "library_stem": library_stem(video_path),
        "model": model,
        "duration_s": round(duration, 2) if duration else None,
        "coverage": round(coverage, 3) if coverage is not None else None,
        "segments": segments,
        "usage": usage_totals,
        "video_probe": probe,
    }


def clean_segments(raw_segments, duration, strict=True):
    """Validate vocabulary, clamp times, sort, resolve overlaps.

    strict=True raises ValueError on out-of-vocab labels (caller retries);
    strict=False coerces them to 'other', keeping the label in notes.
    Returns (segments, coerced_labels).
    """
    coerced = []
    segs = []
    for s in raw_segments:
        move = str(s.get("move", "")).strip().lower()
        if move not in MOVE_VOCAB:
            if strict:
                raise ValueError(f"out-of-vocabulary move label: {move!r}")
            coerced.append(move)
            note = str(s.get("notes", "") or "")
            s["notes"] = f"[original label: {move}] {note}".strip()
            move = "other"
        a, b = float(s.get("start_s", 0)), float(s.get("end_s", 0))
        if duration:
            a, b = max(0.0, min(a, duration)), max(0.0, min(b, duration))
        if b - a < MIN_SEG_SEC:
            continue
        conf = s.get("confidence", None)
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.5
        segs.append({
            "move": move,
            "start_s": round(a, 2),
            "end_s": round(b, 2),
            "confidence": round(conf, 2),
            "notes": str(s.get("notes", "") or ""),
        })

    segs.sort(key=lambda s: (s["start_s"], s["end_s"]))
    out = []
    for s in segs:
        if out and s["start_s"] < out[-1]["end_s"]:
            if s["end_s"] <= out[-1]["end_s"] + OVERLAP_DROP_SEC:
                continue  # fully swallowed by previous segment
            s["start_s"] = out[-1]["end_s"]  # trim to previous end
            if s["end_s"] - s["start_s"] < MIN_SEG_SEC:
                continue
        out.append(s)
    return out, coerced


# ---------------------------------------------------------------- merge

def merge_into(merged_path, result):
    """Upsert one video's tags into the merged JSON, keyed by library stem."""
    merged_path = Path(merged_path)
    if merged_path.exists():
        merged = json.loads(merged_path.read_text(encoding="utf-8"))
    else:
        merged = {
            "_comment": ("Merged VLM move tags keyed by LIBRARY STEM (camera-agnostic, "
                         "matches output/reference_library/aistpp file stems). "
                         "Produced by analyzer_3d/src/vlm_movetag.py; joinable by move_library.py."),
            "clips": {},
        }
    entry = {k: v for k, v in result.items() if k not in ("_comment", "video_probe")}
    merged["clips"][result["library_stem"]] = entry
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"[movetag] merged into {merged_path} ({len(merged['clips'])} clips)")


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="Gemini VLM breaking-move tagger")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output tags JSON (default: <video stem>_movetags.json next to video)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=DEFAULT_SAMPLE_FPS,
                    help="frame-sampling rate Gemini sees (default %(default)s)")
    ap.add_argument("--merge-into", type=Path, default=None,
                    help="merged move_tags.json to upsert this video's tags into (keyed by library stem)")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    out = args.out or args.video.with_name(args.video.stem + "_movetags.json")

    result = tag_video(args.video, model=args.model, sample_fps=args.fps)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[movetag] wrote {out}")

    print(f"\n  {result['library_stem']}  ({result['duration_s']}s, coverage {result['coverage']})")
    for s in result["segments"]:
        print(f"    {s['start_s']:6.2f}-{s['end_s']:6.2f}  {s['move']:<16s} conf={s['confidence']:.2f}  {s['notes']}")

    if args.merge_into:
        merge_into(args.merge_into, result)


if __name__ == "__main__":
    main()
