"""Turn DECO per-vertex contact probabilities into named, time-windowed
contact signatures for the breakdance-coach move library.

Pipeline:
    deco_modal.py  ->  <stem>.deco.npz   (T x 6890 contact probs per frame)
    THIS FILE      ->  contact signatures (per named body part, per window)
                   ->  output/reference_library/contact_signatures.json

Body-part mapping is derived from the SMPL linear-blend-skinning weights
that already ship inside every reference-library pkl (pkl['mesh']['weights'],
shape (6890, 24)). Each vertex is assigned to its dominant joint, and the 24
SMPL joints are grouped into breaking-relevant contact parts (palms, forearms,
feet, knees, head, shoulders, back/glutes). The SMPL topology is identical
across all clips, so one canonical pkl defines the mapping for everything.

A part is "in contact" in a frame when enough of its vertices exceed DECO's
0.5 probability threshold. That boolean timeline is median-filtered to kill
single-frame flicker, then split into contiguous windows.

Usage:
    # Inspect raw part activations for one npz (threshold tuning / debugging)
    python -m analyzer_3d.src.contact_signatures inspect <file.deco.npz>

    # Build a signature for one clip and merge into the master JSON
    python -m analyzer_3d.src.contact_signatures build <file.deco.npz> \
        --stem gBR_sBM_cAll_d04_mBR0_ch09 \
        --merge-into output/reference_library/contact_signatures.json

Output JSON is keyed by LIBRARY STEM (camera-agnostic, same convention as
output/reference_library/move_tags.json).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- part map

# SMPL 24-joint order.
SMPL_JOINTS = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck", "L_collar",
    "R_collar", "head", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hand", "R_hand",
]

# Named contact part -> SMPL joint indices whose dominant vertices belong to it.
# Every joint appears exactly once, so the 6890 vertices partition cleanly.
PART_GROUPS = {
    "palm_L":     [20, 22],       # L_wrist + L_hand
    "palm_R":     [21, 23],       # R_wrist + R_hand
    "forearm_L":  [18],           # L_elbow
    "forearm_R":  [19],           # R_elbow
    "shoulder_L": [13, 16],       # L_collar + L_shoulder (upper arm / shoulder)
    "shoulder_R": [14, 17],
    "foot_L":     [7, 10],        # L_ankle + L_foot
    "foot_R":     [8, 11],
    "knee_L":     [4],
    "knee_R":     [5],
    "head":       [12, 15],       # neck + head
    "back_glutes": [0, 1, 2, 3, 6, 9],  # pelvis + hips + spine chain
}

DEFAULT_CANONICAL_PKL = (
    "output/reference_library/aistpp/gBR_sBM_cAll_d04_mBR0_ch09_smpl_clean.pkl"
)

# --- tuning defaults (validated against pilot ground truth) ---
VERTEX_THR = 0.5        # DECO's own per-vertex contact threshold
PART_MIN_FRAC = 0.10    # >= this fraction of a part's verts must fire ...
PART_MIN_VERTS = 6      # ... AND at least this many verts (small parts)
MEDIAN_WIN = 3          # sampled-frame median filter window (odd)
MERGE_GAP_S = 0.30      # bridge windows separated by less than this
MIN_WINDOW_S = 0.20     # drop windows shorter than this


def build_vertex_parts(canonical_pkl: str) -> np.ndarray:
    """Return (6890,) int array mapping each vertex to a PART_GROUPS index."""
    import pickle

    with open(canonical_pkl, "rb") as f:
        d = pickle.load(f)
    weights = d["mesh"]["weights"]           # (6890, 24)
    dom_joint = weights.argmax(axis=1)       # (6890,) dominant SMPL joint

    joint_to_part = np.full(24, -1, dtype=np.int64)
    part_names = list(PART_GROUPS.keys())
    for pi, pname in enumerate(part_names):
        for j in PART_GROUPS[pname]:
            joint_to_part[j] = pi
    assert (joint_to_part >= 0).all(), "every SMPL joint must map to a part"

    vertex_parts = joint_to_part[dom_joint]  # (6890,)
    return vertex_parts, part_names


def part_activations(contact: np.ndarray, vertex_parts: np.ndarray, n_parts: int):
    """Per-frame per-part (active_bool, confidence).

    active[t,p] = enough of part p's vertices exceed VERTEX_THR at frame t.
    conf[t,p]   = mean prob of that part's firing vertices (0 if none fire).
    """
    T = contact.shape[0]
    active = np.zeros((T, n_parts), dtype=bool)
    conf = np.zeros((T, n_parts), dtype=np.float32)
    contact = contact.astype(np.float32)
    fired = contact >= VERTEX_THR                      # (T, 6890)

    for p in range(n_parts):
        vmask = vertex_parts == p                      # (6890,)
        psize = int(vmask.sum())
        if psize == 0:
            continue
        part_fired = fired[:, vmask]                   # (T, psize)
        counts = part_fired.sum(axis=1)                # (T,)
        need = max(PART_MIN_VERTS, int(np.ceil(PART_MIN_FRAC * psize)))
        active[:, p] = counts >= need
        # confidence = mean prob over the fired verts (per frame)
        pc = contact[:, vmask]
        sums = np.where(part_fired, pc, 0.0).sum(axis=1)
        conf[:, p] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return active, conf


def median_filter_bool(active: np.ndarray, win: int) -> np.ndarray:
    """Temporal median filter over the frame axis (per part)."""
    if win <= 1:
        return active
    T, P = active.shape
    pad = win // 2
    out = np.zeros_like(active)
    a = active.astype(np.float32)
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    for t in range(T):
        out[t] = np.median(ap[t:t + win], axis=0) >= 0.5
    return out


def extract_windows(active_p, conf_p, times, merge_gap, min_dur):
    """Contiguous True runs in a 1-D bool timeline -> list of windows."""
    windows = []
    T = len(active_p)
    t_idx = 0
    while t_idx < T:
        if not active_p[t_idx]:
            t_idx += 1
            continue
        j = t_idx
        while j < T and active_p[j]:
            j += 1
        # run covers sampled frames [t_idx, j-1]
        start_s = float(times[t_idx])
        end_s = float(times[min(j, T - 1)]) if j < T else float(times[-1])
        # extend end to the next sample boundary so a 1-frame run has duration
        if j < T:
            end_s = float(times[j - 1] + (times[j] - times[j - 1]))
        else:
            step = float(times[1] - times[0]) if T > 1 else 0.2
            end_s = float(times[-1] + step)
        mean_conf = float(conf_p[t_idx:j].mean())
        windows.append([start_s, end_s, mean_conf])
        t_idx = j

    # merge windows separated by a small gap
    merged = []
    for w in windows:
        if merged and w[0] - merged[-1][1] <= merge_gap:
            prev = merged[-1]
            # weighted-ish conf merge
            prev[1] = w[1]
            prev[2] = max(prev[2], w[2])
        else:
            merged.append(w[:])
    # drop too-short
    merged = [w for w in merged if (w[1] - w[0]) >= min_dur]
    return [{"start_s": round(w[0], 3), "end_s": round(w[1], 3),
             "mean_conf": round(w[2], 3)} for w in merged]


def summarize(part_windows: dict, part_totals: dict) -> str:
    """Short human sentence describing the dominant contacts."""
    if not part_totals:
        return "no contacts detected"
    ranked = sorted(part_totals.items(), key=lambda kv: -kv[1])
    ranked = [(p, s) for p, s in ranked if s > 0]
    if not ranked:
        return "no contacts detected"
    bits = [f"{p} {s:.1f}s" for p, s in ranked[:6]]
    return "; ".join(bits)


def build_signature(npz_path, vertex_parts, part_names):
    data = np.load(npz_path, allow_pickle=True)
    contact = data["contact"]                # (T, 6890)
    times = data["times"].astype(np.float32)
    sample_fps = float(data["sample_fps"])
    video_fps = float(data["video_fps"])
    crop = bool(data["crop"]) if "crop" in data else None
    had_person = data["had_person"] if "had_person" in data else None

    n_parts = len(part_names)
    active, conf = part_activations(contact, vertex_parts, n_parts)
    active = median_filter_bool(active, MEDIAN_WIN)

    all_windows = []
    part_windows = {}
    part_totals = {}
    for p, pname in enumerate(part_names):
        wins = extract_windows(active[:, p], conf[:, p], times,
                               MERGE_GAP_S, MIN_WINDOW_S)
        if wins:
            part_windows[pname] = wins
            total = sum(w["end_s"] - w["start_s"] for w in wins)
            part_totals[pname] = round(total, 3)
            for w in wins:
                all_windows.append({"part": pname, **w})

    all_windows.sort(key=lambda w: (w["start_s"], w["part"]))
    sig = {
        "fps_sampled": round(sample_fps, 3),
        "video_fps": round(video_fps, 3),
        "num_frames_sampled": int(contact.shape[0]),
        "crop": crop,
        "person_detected_frac": (
            round(float(np.mean(had_person)), 3) if had_person is not None else None
        ),
        "windows": all_windows,
        "part_totals": part_totals,
        "summary": summarize(part_windows, part_totals),
    }
    return sig


def inspect(npz_path, canonical_pkl):
    """Print raw per-part activation stats for threshold tuning."""
    vertex_parts, part_names = build_vertex_parts(canonical_pkl)
    data = np.load(npz_path, allow_pickle=True)
    contact = data["contact"].astype(np.float32)
    times = data["times"]
    n_parts = len(part_names)
    print(f"npz: {npz_path}")
    print(f"  frames={contact.shape[0]} verts={contact.shape[1]} "
          f"sample_fps={float(data['sample_fps'])}")
    print(f"  contact prob: min={contact.min():.3f} max={contact.max():.3f} "
          f"mean={contact.mean():.4f} frac>=0.5={np.mean(contact>=0.5):.4f}")
    if "had_person" in data:
        print(f"  person_detected_frac={np.mean(data['had_person']):.3f}")
    active, conf = part_activations(contact, vertex_parts, n_parts)
    print(f"  {'part':12s} {'active_frames':>13s} {'active_s':>9s} "
          f"{'peak_conf':>9s}")
    step = float(times[1] - times[0]) if len(times) > 1 else 0.2
    for p, pname in enumerate(part_names):
        af = int(active[:, p].sum())
        pc = float(conf[:, p].max())
        print(f"  {pname:12s} {af:>13d} {af*step:>8.2f}s {pc:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect")
    pi.add_argument("npz")
    pi.add_argument("--canonical-pkl", default=DEFAULT_CANONICAL_PKL)

    pb = sub.add_parser("build")
    pb.add_argument("npz")
    pb.add_argument("--stem", required=True, help="library stem key")
    pb.add_argument("--canonical-pkl", default=DEFAULT_CANONICAL_PKL)
    pb.add_argument("--merge-into",
                    default="output/reference_library/contact_signatures.json")
    pb.add_argument("--source-video", default=None)

    args = ap.parse_args()

    if args.cmd == "inspect":
        inspect(args.npz, args.canonical_pkl)
        return

    if args.cmd == "build":
        vertex_parts, part_names = build_vertex_parts(args.canonical_pkl)
        sig = build_signature(args.npz, vertex_parts, part_names)
        if args.source_video:
            sig = {"video": args.source_video, **sig}

        out_path = Path(args.merge_into)
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                merged = json.load(f)
        else:
            merged = {
                "_comment": (
                    "Per-move DENSE CONTACT SIGNATURES keyed by LIBRARY STEM "
                    "(camera-agnostic, matches output/reference_library/aistpp "
                    "stems and move_tags.json). Produced by "
                    "analyzer_3d/src/deco_modal.py + contact_signatures.py. "
                    "Named body-part contact windows from DECO per-vertex "
                    "contact estimation. Body parts derived from SMPL LBS "
                    "dominant-joint mapping."
                ),
                "params": {
                    "vertex_thr": VERTEX_THR,
                    "part_min_frac": PART_MIN_FRAC,
                    "part_min_verts": PART_MIN_VERTS,
                    "median_win": MEDIAN_WIN,
                    "merge_gap_s": MERGE_GAP_S,
                    "min_window_s": MIN_WINDOW_S,
                    "parts": list(PART_GROUPS.keys()),
                },
                "clips": {},
            }
        merged["clips"][args.stem] = sig
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1, ensure_ascii=False)
        print(f"[{args.stem}] {sig['summary']}")
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
