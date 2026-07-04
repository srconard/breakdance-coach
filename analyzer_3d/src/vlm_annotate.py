"""VLM contact-gate annotator.

Gemini watches a clip and emits the coarse contact-phase gates that
contact_refine.py consumes (same JSON contract as the hand-written
flare_demo30_gates.json), plus a clip-suitability triage and an
experimental per-hand plant list.

The gates are deliberately coarse: they only gate WHICH effectors the
2D-keypoint-evidence refinement is allowed to consider planted in a
window. A missed gate kills a real contact downstream, so the prompt
biases Gemini toward slightly-wide windows.

Usage:
    python -m analyzer_3d.src.vlm_annotate video.mp4
        [--out gates.json] [--model gemini-2.5-pro] [--fps 5]
        [--compare refined.contacts.json]

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

# Pro is the default on purpose: on the flare acceptance test flash's
# over-wide feet gate produced 5 false mid-flare foot contacts (R_ankle
# IoU 56% vs reference), while pro matched hand annotation (ankles 100%
# IoU). ~$0.03/clip at 5 fps sampling. Flash remains available via --model.
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SAMPLE_FPS = 5.0
MERGE_GAP_SEC = 0.05  # merge gate intervals separated by less than this


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
        print(f"[vlm] ffprobe failed ({e}); proceeding without local probe")
        return None


# ---------------------------------------------------------------- prompt

def build_prompt(duration, sample_fps, probe):
    dur_txt = f"{duration:.2f} seconds" if duration else "unknown length"
    res_txt = f"{probe['width']}x{probe['height']} @ {probe['fps']:.1f} fps" if probe else "unknown resolution"
    return f"""You are annotating a breakdance/floorwork video for a motion-capture pipeline.
The clip is {dur_txt}, {res_txt}. You see frames sampled at {sample_fps:g} fps, so your timestamps are accurate to roughly +/-{1.0 / sample_fps:.1f}s.

TASK 1 - CONTACT GATES. Report time windows (seconds, one decimal place) where the dancer's HANDS and FEET bear weight on the floor:
- "hands": windows where AT LEAST ONE hand/palm is planted on the floor supporting body weight.
- "feet": windows where AT LEAST ONE foot or its toes rests on the floor (standing, stepping, crouching, launching, landing). IMPORTANT: kneeling and crouching count as foot contact whenever the toes/balls of the feet touch the floor, even if the knees take most of the weight.
Rules:
- These are coarse GATES for a downstream refiner that verifies exact contacts itself: err on the side of slightly WIDER windows (about +/-0.3s) rather than missing any real weight-bearing contact. A missed contact is much worse than a slightly wide window.
- Cover the WHOLE clip, entry through ending: settling into a crouch, kneel or stand at the end still has foot contacts.
- Do NOT include fast skims or brushes where a limb touches the floor without taking weight (e.g. legs sweeping the floor mid-flare, a hand tapping in passing).
- Merge windows of the same type separated by less than about 0.5s.

TASK 2 - PER-HAND DETAIL (experimental). Also list each individual hand plant as {{"side": "left"|"right", "start": s, "end": s}}. "left" means the DANCER'S OWN left hand, not screen-left. During moves like flares the hands alternate rapidly with brief both-hands overlaps; capture that alternation as precisely as you can.

TASK 3 - CLIP TRIAGE for 3D pose reconstruction suitability:
- camera_static: is the camera fixed (tripod-like) for the whole clip?
- full_body_visible: does the dancer's full body stay in frame throughout?
- support_limbs_visible: are the weight-bearing hands/feet visible (not occluded by the body or objects) most of the time?
- clothing_fitted: is clothing fitted enough to read limb pose (vs very baggy)?
- motion_blur: "none" | "mild" | "heavy" (judge the fast-moving limbs)
- single_subject: exactly one person in frame?
- overall: "good" | "marginal" | "bad", with a short "issues" list of concrete problems.

Return STRICT JSON only (no markdown, no commentary), exactly this shape:
{{
  "phases": {{"hands": [[start, end], ...], "feet": [[start, end], ...]}},
  "hands_detail": [{{"side": "left", "start": 0.0, "end": 1.5}}, ...],
  "triage": {{"camera_static": true, "full_body_visible": true, "support_limbs_visible": true, "clothing_fitted": true, "motion_blur": "mild", "single_subject": true, "overall": "good", "issues": []}},
  "notes": "one short paragraph describing the clip structure and what the dancer does"
}}"""


# ---------------------------------------------------------------- gemini

def annotate(video_path, model=DEFAULT_MODEL, sample_fps=DEFAULT_SAMPLE_FPS):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=load_api_key())
    probe = probe_video(video_path)
    duration = probe["duration"] if probe else None

    print(f"[vlm] uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)...")
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
        print(f"[vlm] file ACTIVE after {time.time() - t0:.0f}s; asking {model} (sampling {sample_fps:g} fps)...")

        video_part = types.Part(
            file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type or "video/mp4"),
            video_metadata=types.VideoMetadata(fps=sample_fps),
        )
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[
                video_part,
                types.Part(text=build_prompt(duration, sample_fps, probe)),
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
        print(f"[vlm] tokens: {usage.prompt_token_count}+{usage.candidates_token_count}")

    raw = resp.text or ""
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw)
    data = json.loads(raw)
    if "phases" not in data:
        raise RuntimeError(f"VLM response missing 'phases': {list(data.keys())}")

    for key in ("hands", "feet"):
        data["phases"][key] = clean_intervals(data["phases"].get(key, []), duration)
    for seg in data.get("hands_detail", []):
        seg["start"], seg["end"] = round(float(seg["start"]), 2), round(float(seg["end"]), 2)

    return {
        "_comment": (f"VLM-annotated contact gates ({model}, sampled {sample_fps:g} fps). "
                     "Same contract as hand annotation; consumed by contact_refine.py."),
        "clip": video_path.name,
        "duration": round(duration, 2) if duration else None,
        "phases": data["phases"],
        "hands_detail": data.get("hands_detail", []),
        "triage": data.get("triage", {}),
        "notes": data.get("notes", ""),
        "video_probe": probe,
    }


def clean_intervals(intervals, duration):
    """Clamp to clip, drop degenerate, sort, merge near-touching."""
    segs = []
    for pair in intervals:
        a, b = float(pair[0]), float(pair[1])
        if duration:
            a, b = max(0.0, min(a, duration)), max(0.0, min(b, duration))
        if b - a > 1e-3:
            segs.append([a, b])
    segs.sort()
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] < MERGE_GAP_SEC:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in merged]


# ---------------------------------------------------------------- compare

def _overlap(seg, intervals):
    a, b = seg
    return sum(max(0.0, min(b, i1) - max(a, i0)) for i0, i1 in intervals)


def compare_to_contacts(gates, contacts_path):
    """Acceptance test: do the VLM gates cover the refined per-effector
    contacts?  Coverage <100% on a real contact means the gate would have
    killed it. Also scores per-hand detail for left/right correctness."""
    contacts = json.loads(Path(contacts_path).read_text())
    gate_of = {"L_wrist": "hands", "R_wrist": "hands",
               "L_ankle": "feet", "R_ankle": "feet"}

    print(f"\n=== gate coverage of derived contacts ({Path(contacts_path).name}) ===")
    worst = 1.0
    for eff, segs in contacts.items():
        gate = gates["phases"].get(gate_of[eff], [])
        for a, b in segs:
            cov = _overlap((a, b), gate) / max(b - a, 1e-6)
            worst = min(worst, cov)
            flag = "" if cov >= 0.90 else "  <-- UNDER-COVERED"
            print(f"  {eff:8s} [{a:6.2f},{b:6.2f}] ({b - a:.2f}s)  coverage {cov * 100:5.1f}%{flag}")

    total_contact = {k: sum(b - a for a, b in v) for k, v in contacts.items()}
    hands_contact = total_contact["L_wrist"] + total_contact["R_wrist"]
    feet_contact = total_contact["L_ankle"] + total_contact["R_ankle"]
    hands_gate = sum(b - a for a, b in gates["phases"].get("hands", []))
    feet_gate = sum(b - a for a, b in gates["phases"].get("feet", []))
    print(f"  totals: hands gate {hands_gate:.1f}s vs contact {hands_contact:.1f}s "
          f"(union of L+R, overlap counted twice); feet gate {feet_gate:.1f}s vs contact {feet_contact:.1f}s")
    print(f"  worst segment coverage: {worst * 100:.1f}%  ({'PASS' if worst >= 0.90 else 'FAIL'} at 90% bar)")

    detail = gates.get("hands_detail", [])
    if detail:
        print("\n=== per-hand detail vs derived wrist contacts (experimental) ===")
        for label, mapping in (("as-given", {"left": "L_wrist", "right": "R_wrist"}),
                               ("L/R-swapped", {"left": "R_wrist", "right": "L_wrist"})):
            num = den = 0.0
            for seg in detail:
                ref = contacts.get(mapping.get(seg.get("side", ""), ""), [])
                num += _overlap((seg["start"], seg["end"]), ref)
                den += seg["end"] - seg["start"]
            print(f"  {label:12s}: {num / max(den, 1e-6) * 100:5.1f}% of VLM hand-plant time overlaps the matching wrist's contacts")
    return worst


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="Gemini VLM contact-gate annotator")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output gates JSON (default: <video stem>_gates_vlm.json next to video)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fps", type=float, default=DEFAULT_SAMPLE_FPS,
                    help="frame-sampling rate Gemini sees (default %(default)s)")
    ap.add_argument("--compare", type=Path, default=None,
                    help="refined .contacts.json to score gate coverage against")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"video not found: {args.video}")
    out = args.out or args.video.with_name(args.video.stem + "_gates_vlm.json")

    gates = annotate(args.video, model=args.model, sample_fps=args.fps)
    out.write_text(json.dumps(gates, indent=1), encoding="utf-8")
    print(f"[vlm] wrote {out}")

    tri = gates.get("triage", {})
    print(f"\n  phases.hands: {gates['phases']['hands']}")
    print(f"  phases.feet:  {gates['phases']['feet']}")
    print(f"  hand plants:  {len(gates.get('hands_detail', []))} "
          f"{[(s.get('side', '?')[0].upper(), s['start'], s['end']) for s in gates.get('hands_detail', [])]}")
    print(f"  triage: overall={tri.get('overall', '?')} blur={tri.get('motion_blur', '?')} "
          f"static_cam={tri.get('camera_static', '?')} issues={tri.get('issues', [])}")
    print(f"  notes: {gates.get('notes', '')}")

    if args.compare:
        compare_to_contacts(gates, args.compare)


if __name__ == "__main__":
    main()
