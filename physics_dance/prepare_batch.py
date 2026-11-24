"""Bake every Move Library motion that can go into the physics body, and write
the Modal batch manifests (2026-11-24, physics verdicts, step one).

Runs on the PC from the repo root (Blender for the Mixamo FBX and CMU BVH bakes,
bake_from_pkl for SMPL pkls). Cheap: a few seconds per clip, no network.

    python -m physics_dance.prepare_batch            # bake + manifests
    python -m physics_dance.prepare_batch --dry-run  # list what it would do

Writes output/physics_dance/manifest_<group>.json (groups: ig_trackable,
mocap, ig_usable) for `modal run physics_dance/modal_soma_track.py::batch`.

What goes in (decided in the job brief, 2026-11-24):
  * ig_trackable: the Experiment-1 "trackable now" clips whose 3D is graded clean
    or usable (the four graded broken are skipped: a physics check of a wrong
    3D model would mislead), minus the six already run on 2026-11-17;
  * mocap: every motion-capture reference in the library (Mixamo, CMU 85,
    AIST++, FineDance), one window of at most WINDOW_S seconds each; freestyle
    runs start 1 s before their first floor move (Gemini move tags);
  * ig_usable: the other clean/usable 3D models, except the two that hang from
    a bar (the physics scene has no bar: that is a terrain question, step two).
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

WINDOW_S = 12.0
IG_WINDOW_S = 8.0  # same window as the 2026-11-17 runs, so numbers compare
BLENDER = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
BAKE_SCRIPT = "analyzer_3d/src/blender_scripts/skeleton_bake_npz.py"
OUT = Path("output/physics_dance")
REF = Path("output/reference_library")
IG_PKL = Path("output/move_library/ig_results")

IG_TRACKABLE = ["C4yIajZugSj", "C9pmdf6I0fI", "Cjx2p8CDz2E", "CyomjspuuEJ", "DE11kilPZ95"]
IG_USABLE = ["C1aGb2nNxkx", "C2y9zMCIqJK", "C4yIaj3u2g_", "C9pmdfJokqt", "CgrBahODcqY",
             "DCq7l1XoPVp", "DCq7l8DIctQ", "DCq7mA3ohDX", "DCq7mmKoC-x"]
MIXAMO = ["Flair", "Breakdance 1990", "Breakdance Footwork 3",
          "Breakdance Freezes", "Breakdance Freeze Var 1", "Breakdance Freeze Var 2",
          "Breakdance Freeze Var 3", "Breakdance Freeze Var 4", "Breakdance Ready"]
LOOPS = {"Flair": 3}  # one Mixamo flair cycle is 1 s; the clip is made to loop
CMU = ["85_05", "85_06", "85_07", "85_08", "85_12", "85_14"]  # 85_04 failed the screen (floats, under the floor)
FLOOR_MOVES = {"6-step", "cc", "footwork-other", "baby-freeze", "chair-freeze", "airchair", "elbow-freeze",
               "handstand-freeze", "flare", "windmill", "headspin", "backspin", "swipe", "1990", "2000",
               "halo", "power-combo", "kick-up", "other:floorwork", "other:headslide"}


def frames_of(npz):
    return int(np.load(npz)["JW"].shape[0])


def bake_pkl(pkl, out_npz, dry):
    if out_npz.exists() or dry:
        return out_npz
    subprocess.run(["python", "-m", "physics_dance.bake_from_pkl", str(pkl), "--out", str(out_npz.parent)], check=True)
    made = out_npz.parent / (pkl.name.replace("_smpl_clean.pkl", "") + ".npz")
    if made != out_npz:
        made.replace(out_npz)
    return out_npz


def bake_blender(src, out_npz, preset, dry):
    if out_npz.exists() or dry:
        return out_npz
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([BLENDER, "-b", "--python", BAKE_SCRIPT, "--", str(src), str(out_npz), preset, "30"],
                   check=True, stdout=subprocess.DEVNULL)
    return out_npz


def window(npz, start=0.0, length=WINDOW_S):
    dur = frames_of(npz) / 30.0
    start = max(0.0, min(start, dur - 1.0))
    end = round(min(dur, start + length), 2)
    return {"start": round(start, 2) if start > 0 else None, "end": end}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    bake = OUT / "bake_lib"
    bake.mkdir(parents=True, exist_ok=True)
    tags = json.loads((REF / "move_tags.json").read_text(encoding="utf-8")).get("clips", {})
    groups = {"ig_trackable": [], "mocap": [], "ig_usable": []}

    for grp, ids in (("ig_trackable", IG_TRACKABLE), ("ig_usable", IG_USABLE)):
        for sc in ids:
            pkl = IG_PKL / sc / f"{sc}_smpl_clean.pkl"
            if not pkl.exists():
                print("MISSING", pkl)
                continue
            npz = bake_pkl(pkl, bake / f"ig_{sc}.npz", a.dry_run)
            w = window(npz, 0.0, IG_WINDOW_S) if npz.exists() else {}
            groups[grp].append(dict({"name": f"ig_{sc}", "npz": str(npz).replace("\\", "/"),
                                     "kind": "ig", "ref": sc}, **w))

    for m in MIXAMO:
        npz = bake_blender(Path("downloads/mixamo") / f"{m}.fbx", bake / f"mixamo_{m.replace(' ', '_')}.npz", "mixamo", a.dry_run)
        w = window(npz) if npz.exists() else {}
        groups["mocap"].append(dict({"name": "mx_" + m.replace("Breakdance ", "").replace(" ", "_").lower(),
                                     "npz": str(npz).replace("\\", "/"), "kind": "mocap", "ref": f"{m}_beta", "loops": LOOPS.get(m, 1)}, **w))
    for c in CMU:
        npz = bake_blender(Path("downloads/cmu85") / f"{c}.bvh", bake / f"cmu{c}.npz", "cmu", a.dry_run)
        w = window(npz) if npz.exists() else {}
        groups["mocap"].append(dict({"name": f"cmu{c}", "npz": str(npz).replace("\\", "/"), "kind": "mocap",
                                     "ref": f"{c}_mannequin"}, **w))
    for src in ("aistpp", "finedance"):
        for pkl in sorted((REF / src).glob("*_smpl_clean.pkl")):
            stem = pkl.name.replace("_smpl_clean.pkl", "")
            npz = bake_pkl(pkl, bake / f"{src}_{stem}.npz", a.dry_run)
            segs = (tags.get(stem) or {}).get("segments") or []
            floor = [s for s in segs if s.get("move") in FLOOR_MOVES]
            start = max(0.0, floor[0]["start_s"] - 1.0) if floor else 0.0
            w = window(npz, start) if npz.exists() else {}
            short = ("aist_" + stem.split("_")[1] + "_" + stem.split("_")[-1]) if src == "aistpp" else stem.split("_")[0]
            groups["mocap"].append(dict({"name": short, "npz": str(npz).replace("\\", "/"), "kind": "mocap",
                                         "ref": stem}, **w))

    for g, jobs in groups.items():
        (OUT / f"manifest_{g}.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
        secs = sum((j.get("end") or 0) - (j.get("start") or 0) for j in jobs)
        print(f"{g}: {len(jobs)} runs, {secs:.0f} s of motion")
        for j in jobs:
            print("  ", j["name"], j.get("start"), j.get("end"))


if __name__ == "__main__":
    main()
