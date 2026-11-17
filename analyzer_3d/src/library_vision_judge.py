"""Vision pass over the move-library audit sheets (2026-11-17).

Shows each `<id>_audit.jpg` (made by library_audit.py: row 1 = source frames
with the 3D skeleton projected on top, row 2 = the world-frame 3D at the same
instants with the floor line) to a vision model through the Codex CLI — $0 on
Shawn's ChatGPT plan, no API key — and asks for a clean / usable / broken
verdict with failure tags. Writes `<id>_vision.json` next to each sheet and
`vision_summary.json`.

The model only sees pictures; the numbers in `<id>_audit.json` are passed as
context so it can weigh them, and the final grade is decided by
move_library_site.py from both (see `final_grade`).

Usage (PC lane):
    python -m analyzer_3d.src.library_vision_judge AUDIT_DIR [--only ID ...]
        [--workers 3] [--model gpt-5.6-sol]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TAGS = ["floating", "missing_contact", "foot_sliding", "wrong_limb", "drift",
        "occlusion", "fast_cuts", "no_rotation", "leg_underextension", "jitter",
        "pose_mismatch", "none"]

PROMPT = """You are auditing a 3D human-motion reconstruction made from a single-camera video.
The image is an audit sheet. ROW 1: frames of the source video with the reconstructed
3D skeleton projected on top (teal = body LEFT, orange = body RIGHT; white dots = head,
wrists, ankles). ROW 2: the same 3D reconstruction at the same instants in world space,
viewed from the camera's heading, gravity-aligned; the grey horizontal line is the floor.
Labels say the time and the lowest body point's height; red "FLOATING" means no body
part has been within 5 cm of the floor for 0.5 s or more.

Measured numbers for this clip (from the reconstruction itself): {numbers}

Judge how well the 3D matches the real movement, as a dancer who wants to study the
move in slow motion from other angles would. Check: does the skeleton in row 1 sit on
the person's limbs; does row 2 show the same pose; are hands/feet/back/head ON the
floor in row 2 when they are on the floor in the video (floating = broken for floor
moves); are left/right limbs swapped; does the body drift or slide; are there cuts,
text overlays, close-ups or occlusion that the reconstruction cannot handle.

Grades: "clean" = could study the move from it as-is; "usable" = right move and
shape, with visible defects a learner can see past; "broken" = misleading (wrong pose,
floating through a floor move, wrong limbs, garbage segments dominate).

Answer ONLY with one JSON object, no prose:
{{"grade": "clean|usable|broken", "tags": [subset of {tags}],
 "reason": "<= 30 words, plain English, the main thing wrong or right",
 "pose_match_row1": 0-10, "world_plausible_row2": 0-10}}"""


def judge_one(sheet, numbers, model, timeout=420):
    prompt = PROMPT.format(numbers=json.dumps(numbers), tags=TAGS)
    with tempfile.TemporaryDirectory() as td:
        last = Path(td) / "last.txt"
        exe = shutil.which("codex") or "codex"
        cmd = [exe, "exec", "--skip-git-repo-check", "-s", "read-only",
               "-i", str(Path(sheet).resolve()), "--output-last-message", str(last)]
        if model:
            cmd += ["-m", model]
        cmd += ["-"]          # prompt on stdin: no shell quoting of a multi-line prompt
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                           cwd=td, encoding="utf-8", errors="replace")
        text = last.read_text(encoding="utf-8", errors="replace") if last.exists() else r.stdout
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"error": "no JSON", "raw": text[-400:], "rc": r.returncode}
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return {"error": f"bad JSON: {e}", "raw": m.group(0)[:400]}
    out["judge"] = f"codex exec {model or 'default'} (vision, $0)"
    return out


NUM_KEYS = ["duration_s", "reproj_med", "float_s", "feet_off_s", "float_frac",
            "slide_cm_s", "jitter_deg_s2", "valid_frac", "n_cycles", "travel_m_per_s",
            "contact_summary"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audit_dir")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    d = Path(a.audit_dir)
    jobs = []
    for sheet in sorted(d.glob("*_audit.jpg")):
        cid = sheet.name[:-len("_audit.jpg")]
        if a.only and cid not in a.only:
            continue
        outp = d / f"{cid}_vision.json"
        if outp.exists() and not a.redo:
            continue
        ap_ = d / f"{cid}_audit.json"
        if not ap_.exists():      # sheet written, numbers not yet: next run
            continue
        aud = json.loads(ap_.read_text())
        jobs.append((cid, sheet, {k: aud.get(k) for k in NUM_KEYS}, outp))

    def run(job):
        cid, sheet, nums, outp = job
        try:
            v = judge_one(sheet, nums, a.model)
        except subprocess.TimeoutExpired:
            v = {"error": "timeout"}
        outp.write_text(json.dumps(v, indent=1))
        print(f"[vision] {cid}: {v.get('grade', 'ERR')} {v.get('tags', '')} {v.get('reason', v.get('error', ''))}",
              flush=True)
        return cid, v

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run, jobs))
    summary = {}
    for p in sorted(d.glob("*_vision.json")):
        summary[p.name[:-len("_vision.json")]] = json.loads(p.read_text())
    (d / "vision_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
