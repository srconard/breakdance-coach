"""Physics verdicts for the Move Library (2026-11-24, physics movement project step one).

Reads every run under output/physics_dance/runs/<name>/ (metrics.json from
render_compare.py, re-scored with lost_at_s) plus the batch manifests, and
writes output/physics_dance/library/runs.json + a hard copy of each
side-by-side video, ready to copy to the NAS (ig-extract/library-assets/physics/).

    python -m physics_dance.verdicts            # PC, repo root; stdlib only

THE THREE LABELS (thresholds from the 2026-11-17 numbers: NVIDIA's own clips
4.6 cm, standing toprock 13 cm and never falls, popping 9-10 cm, floor moves
lost by 0.5-0.9 s):
  * "holds up in physics": the body never fell and never lost the move
    (lost = its 1-second average error passed 25 cm, or its pelvis ended up
    more than 50 cm from where the move puts it), over the whole window;
  * "partly (loses it at N s)": it lost the move at N >= 1.5 s;
  * "falls apart": it lost the move in the first 1.5 s.
A verdict says only what THIS pretrained body (NVIDIA's anatomical SOMA
tracker, trained on everyday and sports motion) did with THIS motion. It is
never a claim about what a human can do.
"""
import json
import shutil
from pathlib import Path

RUNS = Path("output/physics_dance/runs")
OUT = Path("output/physics_dance/library")
LOST_FAST_S = 1.5
HOLD_RULE = ("Holds up = never fell and never lost the move. Lost = the body's 1-second average distance from the "
             "move passed 25 cm, or its hips ended up more than 50 cm from where the move puts them. "
             "Falls apart = lost in the first 1.5 s. Partly = lost later.")
FLOOR_FAMILIES_HINT = ("Hands-on-the-floor breaking is outside what this body learned (it trained on everyday and "
                       "sports motion), so this is a training gap, not proof the move is impossible.")

# The 2026-11-17 runs (first test): name -> (kind, ref, start, end, note). Imported, not re-run.
FIRST_TEST = {
    "control_mini": ("control", "nvidia_mini", None, None, "NVIDIA's own test clips: the control. 4.6 cm is the body at its best."),
    "uprock": ("mocap", "Breakdance Uprock_beta", None, None, "Mixamo uprock (standing toprock)."),
    "sixstep_fw1": ("mocap", "Breakdance Footwork 1_beta", None, None, "Mixamo Footwork 1: two six-step circles."),
    "sixstep_fw1_half": ("variant", "Breakdance Footwork 1_beta", None, None, "The same six-step at half speed."),
    "sixstep_fw1_lift3": ("variant", "Breakdance Footwork 1_beta", None, None, "The same six-step lifted 3 cm off the floor."),
    "sixstep_fw2": ("mocap", "Breakdance Footwork 2_beta", None, None, "Mixamo Footwork 2: one six-step circle."),
    "ig_C6lkJCSrBpn": ("ig", "C6lkJCSrBpn", None, 8.0, ""),
    "ig_DGEOgxHIzef": ("ig", "DGEOgxHIzef", None, 8.0, ""),
    "ig_DR4kebEiPK_": ("ig", "DR4kebEiPK_", None, 8.0, ""),
    "ig_DI6kMAxMEch": ("ig", "DI6kMAxMEch", None, 8.0, "Walks in and stands well, then falls the moment it goes to the floor."),
    "ig_DAi11AvyT93": ("ig", "DAi11AvyT93", None, 8.0, ""),
    "ig_C97qsA0I5ZG": ("ig", "C97qsA0I5ZG", None, 8.0, ""),
}


def verdict(m):
    lost = m.get("lost_at_s")
    if lost is None and m.get("fell_at_s") is not None:
        lost = m["fell_at_s"]
    if lost is None:
        return "holds", "holds up in physics", None
    if lost < LOST_FAST_S:
        return "falls", "falls apart", lost
    return "partly", f"partly (loses it at {lost:g} s)", lost


def why_line(kind, label, m, lost):
    if label == "holds":
        return f"Stays within {m['mean_body_err_m'] * 100:.0f} cm on average and never falls."
    pelvis = m.get("pelvis_z_sim_mean"), m.get("pelvis_z_ref_mean")
    low = pelvis[0] is not None and pelvis[1] is not None and pelvis[0] < pelvis[1] - 0.15
    s = f"Lost the move at {lost:g} s"
    if m.get("fell_at_s") is not None:
        s += f" (fell at {m['fell_at_s']:g} s)"
    s += "; the body sank lower than the move." if low else "."
    if kind == "ig":
        s += " The 3D from video may also be off; see its 3D badge."
    return s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {}
    for mf in sorted(Path("output/physics_dance").glob("manifest_*.json")):
        for j in json.loads(mf.read_text(encoding="utf-8")):
            plan[j["name"]] = dict(j, group=mf.stem.replace("manifest_", ""))
    runs = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        if d.name in FIRST_TEST:
            kind, ref, start, end, note = FIRST_TEST[d.name]
            date, group = "2026-11-17", "first_test"
        elif d.name in plan:
            p = plan[d.name]
            kind, ref, start, end, note = p["kind"], p["ref"], p.get("start"), p.get("end"), ""
            date, group = "2026-11-24", p["group"]
        else:
            continue
        code, label, lost = verdict(m)
        vid = d / f"{d.name}.mp4"
        if vid.exists():
            shutil.copy2(vid, OUT / f"{d.name}.mp4")
        runs.append({
            "name": d.name, "kind": kind, "ref": ref, "start": start, "end": end, "loops": (plan.get(d.name) or {}).get("loops", 1),
            "date": date, "group": group, "verdict": code, "label": label, "lost_at_s": lost,
            "mean_cm": round(m["mean_body_err_m"] * 100, 1), "max_cm": round(m["max_body_err_m"] * 100, 1),
            "fell_at_s": m.get("fell_at_s"), "seconds": m.get("seconds"),
            "pct_within_15cm": m.get("pct_steps_err_lt_0.15m"),
            "err_10hz_cm": [round(x * 100, 1) for x in m.get("err_10hz_m") or []],
            "video": f"{d.name}.mp4" if vid.exists() else None,
            "why": why_line(kind, code, m, lost), "note": note,
        })
    doc = {"_comment": __doc__.split("\n\n")[0], "built": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
           "rules": HOLD_RULE, "floor_hint": FLOOR_FAMILIES_HINT,
           "body": "NVIDIA anatomical SOMA BONES-SEED FSQ tracker (ProtoMotions 13de0cc), MuJoCo, flat floor, Modal CPU",
           "runs": runs}
    (OUT / "runs.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    counts = {}
    for r in runs:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print(f"{r['name']:<24} {r['label']:<30} mean {r['mean_cm']:>5} cm  max {r['max_cm']:>5}  {r['group']}")
    print("COUNTS", json.dumps(counts), len(runs))


if __name__ == "__main__":
    main()
