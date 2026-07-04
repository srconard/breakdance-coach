"""VLM output judge — the third Gemini role from the research decision.

Gemini watches a side-by-side comparison (LEFT: real video, RIGHT: rendered
3D reconstruction) and returns structured per-phase scores + timestamped
defects, so nightly runs can self-grade output without a human frame-stepping.

Companions: vlm_annotate.py (role 1: contact gates, role 2: clip triage).

Usage:
    # judge an existing side-by-side comparison video
    python -m analyzer_3d.src.vlm_judge --compare compare.mp4

    # or build the side-by-side first from source + render
    python -m analyzer_3d.src.vlm_judge --source clip.mp4 --render preview.mp4

Writes <input stem>_judgment.json next to the input.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .vlm_annotate import load_api_key, probe_video

# Pro for judging: the flash-vs-pro gate bake-off showed flash lacks the
# temporal/spatial acuity this kind of call needs.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SAMPLE_FPS = 5.0


def build_side_by_side(source, render, out_path):
    """hstack source (scaled to 800px height) with a 640x800 render preview."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error",
         "-i", str(source), "-i", str(render),
         "-filter_complex",
         "[0:v]fps=30,scale=-2:800[a];[1:v]fps=30[b];[a][b]hstack=shortest=1",
         "-c:v", "libx264", "-crf", "23", "-an", str(out_path)],
        check=True, timeout=300)
    return Path(out_path)


def build_prompt(duration, sample_fps):
    dur_txt = f"{duration:.1f} seconds" if duration else "unknown length"
    return f"""This video is a side-by-side comparison, {dur_txt}, frames sampled at {sample_fps:g} fps (timestamps accurate to ~+/-{1.0 / sample_fps:.1f}s).
LEFT: real video of a dancer performing breakdance/floorwork.
RIGHT: a 3D reconstruction of the same motion (untextured gray body model rendered on a checkered floor, camera angle may differ slightly from the real footage).

You are judging whether the reconstruction is accurate enough for a student to STUDY THE MOVE from the 3D model. Judge the MOTION, not the rendering style: the gray featureless body, missing clothes/face/props, and viewpoint differences are all expected — do not penalize them.

TASK 1 - PHASES. Segment the clip into movement phases (e.g. entry, main move, exit — whatever fits) with start/end timestamps.

TASK 2 - PER-PHASE SCORES, each 0-10 (10 = flawless):
- contact_grounding: in the render, are hands/feet planted ON the floor whenever they bear weight in the real video? Floating or hovering support limbs and floor penetration are the failures to catch.
- pose_fidelity: does the body configuration match (leg height/split, arm positions, torso lean, inversion)?
- placement_stability: does the rendered body stay planted in place the way the real dancer does (no sliding, drifting, or teleporting)?
- motion_quality: timing and smoothness match (no jitter, stutter, or lag).

TASK 3 - DEFECTS. List every concrete problem you can see: {{"time": seconds, "phase": name, "description": what is wrong, "severity": "minor"|"moderate"|"severe"}}. Be specific ("left leg 40 degrees too low", "support hand hovers above floor", "body slides backward").

TASK 4 - OVERALL: {{"score": 0-10, "verdict": "study_ready"|"usable_with_caveats"|"not_usable", "summary": one paragraph}}.

Return STRICT JSON only (no markdown), exactly:
{{
  "phases": [{{"name": "entry", "start": 0.0, "end": 1.2, "scores": {{"contact_grounding": 8, "pose_fidelity": 7, "placement_stability": 9, "motion_quality": 8}}}}, ...],
  "defects": [{{"time": 2.0, "phase": "main", "description": "...", "severity": "moderate"}}, ...],
  "overall": {{"score": 7, "verdict": "usable_with_caveats", "summary": "..."}}
}}"""


def judge(video_path, model=DEFAULT_MODEL, sample_fps=DEFAULT_SAMPLE_FPS):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    probe = probe_video(video_path)
    duration = probe["duration"] if probe else None

    print(f"[judge] uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)...")
    f = client.files.upload(file=str(video_path))
    try:
        t0 = time.time()
        while f.state and f.state.name == "PROCESSING":
            if time.time() - t0 > 300:
                raise RuntimeError("Gemini file processing timed out after 300s")
            time.sleep(2)
            f = client.files.get(name=f.name)
        if not f.state or f.state.name != "ACTIVE":
            raise RuntimeError(f"upload ended in state {f.state.name if f.state else 'unknown'}")
        print(f"[judge] file ACTIVE after {time.time() - t0:.0f}s; asking {model}...")

        video_part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type or "video/mp4"),
            video_metadata=types.VideoMetadata(fps=sample_fps),
        )
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[
                video_part,
                types.Part(text=build_prompt(duration, sample_fps)),
            ])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )
    finally:
        try:
            client.files.delete(name=f.name)
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass

    usage = resp.usage_metadata
    if usage:
        print(f"[judge] tokens: {usage.prompt_token_count}+{usage.candidates_token_count}")

    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", resp.text or "")
    data = json.loads(raw)
    for key in ("phases", "overall"):
        if key not in data:
            raise RuntimeError(f"judge response missing '{key}': {list(data.keys())}")
    data["_meta"] = {"model": model, "sample_fps": sample_fps, "clip": video_path.name}
    return data


def print_judgment(j):
    print(f"\n  overall: {j['overall'].get('score', '?')}/10  [{j['overall'].get('verdict', '?')}]")
    for ph in j.get("phases", []):
        s = ph.get("scores", {})
        print(f"  {ph.get('name', '?'):12s} [{ph.get('start', 0):5.1f}-{ph.get('end', 0):5.1f}s]  "
              f"contact={s.get('contact_grounding', '?')} pose={s.get('pose_fidelity', '?')} "
              f"place={s.get('placement_stability', '?')} motion={s.get('motion_quality', '?')}")
    for d in j.get("defects", []):
        print(f"  ! t={d.get('time', '?'):>5} [{d.get('severity', '?'):8s}] {d.get('description', '')}")
    print(f"  summary: {j['overall'].get('summary', '')}")


def main():
    ap = argparse.ArgumentParser(description="Gemini VLM output judge")
    ap.add_argument("--compare", type=Path, help="existing side-by-side comparison video")
    ap.add_argument("--source", type=Path, help="real video (used with --render)")
    ap.add_argument("--render", type=Path, help="rendered preview (used with --source)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=DEFAULT_SAMPLE_FPS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        video = args.compare
    elif args.source and args.render:
        video = args.source.with_name(args.source.stem + "_vs_render.mp4")
        print(f"[judge] building side-by-side: {video.name}")
        build_side_by_side(args.source, args.render, video)
    else:
        sys.exit("need --compare, or --source and --render")
    if not video.exists():
        sys.exit(f"video not found: {video}")

    j = judge(video, model=args.model, sample_fps=args.fps)
    out = args.out or video.with_name(video.stem + "_judgment.json")
    out.write_text(json.dumps(j, indent=1), encoding="utf-8")
    print(f"[judge] wrote {out}")
    print_judgment(j)


if __name__ == "__main__":
    main()
