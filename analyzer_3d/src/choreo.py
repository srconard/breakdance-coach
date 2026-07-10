"""Choreography composer (v0, kinematic) — chain move segments from library
SMPL pkls into one dance that travels through a scene.

Takes a JSON choreography spec (ordered move segments + where each should
happen), rigidly retargets each segment (yaw + XZ) so the chain is
position-continuous, crossfades the seams in quaternion space, optionally
injects a smooth root drift so a segment *travels* to a target point
("traveling toprock" — library clips dance nearly in place, net root motion
0.05-0.9 m), and exports through the existing mannequin GLB + viewer path.

v1 adds motion-matched seams ("transitions": "matched"): instead of blending
0.4s at the fixed spec boundaries, each seam searches the last ~1.5s of the
outgoing segment x the first ~1.5s of the incoming segment for the cut pair
with minimum motion-matching feature distance (see motion_match.py — pose +
velocity features in the yaw-normalized root frame, Kovar-style windowed
distance), trims both clips at the matched frames, and blends only ~0.15s.
Features are invariant to the rigid yaw+XZ retarget, so matching runs before
alignment and the travel-drift / station-chaining logic is unchanged.

This is the engine for the future sequence-builder UI: the spec is the
interface — a UI only needs to emit the JSON.

Honest v0 limits (kinematic, not physics): injected travel slides the feet
(plausible below ~1.0 m/s during stepping moves, warned above 1.2); seam
poses are nlerp/slerp blends, not motion-matched; no terrain adaptation —
choreograph on open ground (a physics controller, MaskedMimic/GPC-class,
is the P3 path to terrain-valid movement).

Spec JSON:
{
  "title": "Dance through the test scene",
  "scene": "output/reference_library/qa/test_scene.glb",   # optional env GLB
  "transitions": "crossfade",       # or "matched" (motion-matched seam cuts)
  "blend_s": 0.4,                   # seam crossfade (crossfade mode)
  "matched_blend_s": 0.15,          # short blend at matched cuts (matched mode)
  "start": {"at": [0.0, -4.4], "heading_deg": 0},
  "steps": [
    {"label": "entrance toprock",
     "clip": "output/reference_library/aistpp/gBR_..._smpl_clean.pkl",
     "start_s": 0.0, "end_s": 5.5,        # optional trim (clip seconds)
     "heading_deg": 15,                    # optional absolute entry yaw
     "face_travel": true,                  # heading = direction toward `to`
     "to": [x, z]}                         # optional: exit lands here (drift)
  ]
}

Heading convention: 0 deg = facing +Z, degrees CCW around +Y (world).

Usage:
    python -m analyzer_3d.src.choreo spec.json [-o output/choreo/name]
        [--dry-run] [--no-viewer] [--transitions crossfade|matched] [--ab-note]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from .make_viewer import make_viewer
from .mannequin import export_mannequin
from .motion_cleanup import aa_to_quat, fk, mat_to_aa, quat_slerp, quat_to_aa, rodrigues
from .motion_match import extract_features, feature_stats, find_transition

UP = np.array([0.0, 1.0, 0.0])
SPEED_WARN = 1.2      # m/s of *injected* drift that starts to read as sliding
HEAD_WIN_S = 0.5      # entry/exit heading averaged over this window
MATCH_SEARCH_S = 1.5  # matched mode: seam search window on each side of the cut
MATCH_WIN_S = 0.15    # matched mode: +/- window for the windowed feature distance
MATCH_BLEND_S = 0.15  # matched mode: short blend at the matched cut


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def smoothstep(t):
    return t * t * (3 - 2 * t)


def heading_of(jw):
    """Per-frame yaw of the hips, radians; 0 = facing +Z, CCW about +Y."""
    lat = jw[:, 1, :] - jw[:, 2, :]          # L_hip - R_hip
    fwd = np.cross(lat, UP[None, :])
    return np.arctan2(fwd[:, 0], fwd[:, 2])


def mean_heading(h):
    return np.arctan2(np.sin(h).mean(), np.cos(h).mean())


class Step:
    def __init__(self, spec, rest):
        self.spec = spec
        self.label = spec.get("label", Path(spec["clip"]).stem)
        with open(spec["clip"], "rb") as f:
            pkl = pickle.load(f)
        g = pkl["smpl_params_global"]
        self.fps = float(pkl["fps"])
        self.mesh = pkl["mesh"]
        f0 = int(round(spec.get("start_s", 0.0) * self.fps))
        f1 = int(round(spec["end_s"] * self.fps)) if "end_s" in spec else len(g["transl"])
        self.bp = g["body_pose"][f0:f1].astype(np.float64)
        self.go = g["global_orient"][f0:f1].astype(np.float64)
        self.tr = g["transl"][f0:f1].astype(np.float64)
        self.rest = rest if rest is not None else self.mesh["joints"].astype(np.float64)
        self._refresh()

    def _refresh(self):
        jw, _, _ = fk(self.bp, self.go, self.tr, self.rest)
        self.pelvis = jw[:, 0, :]
        self.heading = heading_of(jw)

    @property
    def n(self):
        return len(self.tr)

    def trim(self, start=0, end=None):
        """Drop frames outside [start:end) from every per-frame array."""
        for name in ("bp", "go", "tr", "pelvis", "heading"):
            setattr(self, name, getattr(self, name)[start:end])

    def head_win(self):
        return max(1, int(round(HEAD_WIN_S * self.fps)))

    def entry_heading(self):
        return mean_heading(self.heading[: self.head_win()])

    def align(self, anchor_xz, desired_heading):
        """Rigid yaw+XZ retarget: entry pelvis -> anchor, entry heading -> desired."""
        theta = desired_heading - self.entry_heading()
        R = rot_y(theta)
        # Rotate global orientation.
        go_R = rodrigues(self.go[:, None, :])[:, 0]          # (T,3,3)
        self.go = mat_to_aa(R[None] @ go_R)
        # Desired pelvis trajectory: rotate about entry pelvis (XZ), move to anchor.
        pivot = self.pelvis[0] * np.array([1.0, 0.0, 1.0])
        target = (R @ (self.pelvis - pivot).T).T + pivot
        target[:, 0] += anchor_xz[0] - pivot[0]
        target[:, 2] += anchor_xz[1] - pivot[2]
        # FK is exactly linear in transl: solve transl per frame.
        jw0, _, _ = fk(self.bp, self.go, np.zeros_like(self.tr), self.rest)
        self.tr = target - jw0[:, 0, :]
        self.pelvis = target
        self.heading = self.heading + theta

    def inject_travel(self, to_xz):
        """Smoothstep root drift so the exit pelvis lands on to_xz. Returns drift m, m/s."""
        gap = np.array([to_xz[0] - self.pelvis[-1, 0], to_xz[1] - self.pelvis[-1, 2]])
        dist = float(np.linalg.norm(gap))
        s = smoothstep(np.linspace(0.0, 1.0, self.n))
        drift = np.zeros((self.n, 3))
        drift[:, 0] = gap[0] * s
        drift[:, 2] = gap[1] * s
        self.tr += drift
        self.pelvis += drift
        return dist, dist / (self.n / self.fps)


def crossfade(prev, nxt, overlap):
    """Blend prev's last `overlap` frames with nxt's first `overlap` frames.
    Returns (bp, go, tr) arrays for the blended window."""
    w = smoothstep(np.linspace(0.0, 1.0, overlap))[:, None]
    q_a = aa_to_quat(prev.bp[-overlap:].reshape(overlap, 21, 3))
    q_b = aa_to_quat(nxt.bp[:overlap].reshape(overlap, 21, 3))
    q_b *= np.sign((q_a * q_b).sum(-1, keepdims=True))      # hemisphere align
    bp = quat_to_aa(quat_slerp(q_a, q_b, w[:, :, None])).reshape(overlap, 63)
    qg_a = aa_to_quat(prev.go[-overlap:])
    qg_b = aa_to_quat(nxt.go[:overlap])
    qg_b *= np.sign((qg_a * qg_b).sum(-1, keepdims=True))
    go = quat_to_aa(quat_slerp(qg_a, qg_b, w))
    tr = (1 - w) * prev.tr[-overlap:] + w * nxt.tr[:overlap]
    return bp, go, tr


def match_seams(steps, overlap, fps):
    """Motion-matched seam cuts (v1). For each seam, search the outgoing
    segment's last MATCH_SEARCH_S x the incoming segment's first
    MATCH_SEARCH_S for the cut pair with minimum windowed feature distance,
    then trim both steps at the matched frames (in place). Features are
    invariant to the rigid yaw+XZ retarget, so this runs *before* align()
    and the drift/station chaining downstream is untouched.

    Returns per-seam report rows:
        (label, drop_a_s, drop_b_s, dist_matched, dist_fixed)
    where drop_*_s are seconds trimmed off A's tail / B's head and
    dist_fixed is the feature distance of the v0 fixed-boundary cut."""
    search = int(round(MATCH_SEARCH_S * fps))
    win = int(round(MATCH_WIN_S * fps))
    feats = []
    for st in steps:
        jw, _, _ = fk(st.bp, st.go, st.tr, st.rest)
        feats.append(extract_features(jw, fps))
    mean, std = feature_stats(np.concatenate(feats))
    feats = [(f - mean) / std for f in feats]

    rows = []
    for k in range(len(steps) - 1):
        A, B = steps[k], steps[k + 1]
        label = f"{A.label} -> {B.label}"
        exit_lo = max(overlap - 1, A.n - search)   # keep >= overlap frames in A
        entry_hi = min(search, B.n - overlap)      # keep >= overlap frames in B
        if exit_lo >= A.n - 1 or entry_hi <= 1:
            print(f"[choreo] WARN {label}: segment too short to search seam, "
                  f"keeping fixed boundary")
            rows.append((label, 0.0, 0.0, None, None))
            continue
        i, j, d, D = find_transition(feats[k], feats[k + 1],
                                     exit_lo, A.n, 0, entry_hi, win)
        rows.append((label, (A.n - 1 - i) / fps, j / fps, d, float(D[-1, 0])))
        A.trim(end=i + 1)
        feats[k] = feats[k][:i + 1]
        B.trim(start=j)
        feats[k + 1] = feats[k + 1][j:]
    return rows


def write_ab_note(out_dir, spec, spec_path, seams, seam_ts, overlap, fps):
    """A/B evidence note: per-seam matched vs fixed-boundary feature distance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seam A/B — motion-matched vs fixed-boundary transitions",
        "",
        f"Spec: `{spec_path}` ({spec.get('title', '')})",
        f"Matched mode: search {MATCH_SEARCH_S}s x {MATCH_SEARCH_S}s per seam, "
        f"windowed distance +/-{MATCH_WIN_S}s, blend {overlap / fps:.2f}s "
        f"(v0 baseline: {spec.get('blend_s', 0.4)}s crossfade at the fixed "
        f"spec boundaries).",
        "",
        "Distances are RMS z-scored motion-matching feature distances "
        "(motion_match.py); lower = smoother cut. `fixed` is the distance of "
        "the cut v0 makes; `drop A/B` = seconds trimmed off the outgoing "
        "tail / incoming head to reach the matched cut.",
        "",
        "| seam | t (s) | drop A (s) | drop B (s) | matched | fixed | gain |",
        "|---|---|---|---|---|---|---|",
    ]
    for t_seam, (label, da, db, d, d_fixed) in zip(seam_ts, seams):
        if d is None:
            lines.append(f"| {label} | {t_seam:.1f} | - | - | - | - | "
                         "fixed boundary kept |")
        else:
            gain = (1 - d / max(d_fixed, 1e-9)) * 100
            lines.append(f"| {label} | {t_seam:.1f} | {da:.2f} | {db:.2f} | "
                         f"{d:.3f} | {d_fixed:.3f} | {gain:+.0f}% |")
    note = out_dir / "transitions_ab.md"
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[choreo] wrote {note}")


def compose(spec_path, out_dir=None, dry_run=False, no_viewer=False, transitions=None,
            ab_note=False):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir or Path("output/choreo") / Path(spec_path).stem)
    steps = [Step(s, None) for s in spec["steps"]]
    rest = steps[0].rest
    fps = steps[0].fps
    for st in steps:
        assert st.fps == fps, f"fps mismatch: {st.label}"
    mode = transitions or spec.get("transitions", "crossfade")
    if mode not in ("crossfade", "matched"):
        raise ValueError(f"unknown transitions mode: {mode}")
    blend_s = (spec.get("matched_blend_s", MATCH_BLEND_S) if mode == "matched"
               else spec.get("blend_s", 0.4))
    overlap = int(round(blend_s * fps))
    seams = match_seams(steps, overlap, fps) if mode == "matched" else None

    anchor = np.array(spec["start"]["at"], dtype=float)
    heading = np.radians(spec["start"].get("heading_deg", 0.0))
    rows = []
    for i, st in enumerate(steps):
        s = st.spec
        if s.get("face_travel") and "to" in s:
            heading = float(np.arctan2(s["to"][0] - anchor[0], s["to"][1] - anchor[1]))
        if "heading_deg" in s:
            heading = np.radians(float(s["heading_deg"]))
        st.align(anchor, heading)
        drift = st.inject_travel(s["to"]) if "to" in s else (0.0, 0.0)
        if drift[1] > SPEED_WARN:
            print(f"[choreo] WARN {st.label}: injected drift {drift[1]:.2f} m/s "
                  f"(> {SPEED_WARN}) will read as sliding")
        # Next step anchors at the start of this step's seam window.
        j = max(0, st.n - overlap) if i < len(steps) - 1 else st.n - 1
        anchor = st.pelvis[j][[0, 2]]
        heading = mean_heading(st.heading[max(0, j - st.head_win()): j + 1])
        rows.append((st.label, st.n, st.pelvis[0], st.pelvis[-1],
                     np.degrees(st.entry_heading()), *drift))

    print(f"\n[choreo] {spec.get('title', Path(spec_path).stem)} — "
          f"{len(steps)} steps @ {fps:.0f} fps, {mode} seams, blend {overlap} f")
    print(f"{'label':<28}{'frames':>7}{'entry xz':>16}{'exit xz':>16}"
          f"{'yaw':>7}{'drift':>7}{'m/s':>6}")
    for label, n, p0, p1, yaw, dm, ds in rows:
        print(f"{label:<28}{n:>7}{p0[0]:>8.1f},{p0[2]:>6.1f}"
              f"{p1[0]:>9.1f},{p1[2]:>6.1f}{yaw:>7.0f}{dm:>7.2f}{ds:>6.2f}")

    if seams is not None:
        seam_ts, total = [], steps[0].n
        for k in range(len(seams)):
            seam_ts.append((total - overlap / 2) / fps)
            total += steps[k + 1].n - overlap
        print(f"\n[choreo] matched seams (search {MATCH_SEARCH_S}s x "
              f"{MATCH_SEARCH_S}s, window +/-{MATCH_WIN_S}s; RMS feature "
              f"dist, lower = smoother; fixed = v0 boundary cut)")
        print(f"{'seam':<52}{'t(s)':>6}{'dropA':>7}{'dropB':>7}"
              f"{'matched':>9}{'fixed':>7}{'gain':>6}")
        for t_seam, (label, da, db, d, d_fixed) in zip(seam_ts, seams):
            if d is None:
                print(f"{label:<52}{t_seam:>6.1f}   (fixed boundary kept)")
            else:
                gain = (1 - d / max(d_fixed, 1e-9)) * 100
                print(f"{label:<52}{t_seam:>6.1f}{da:>7.2f}{db:>7.2f}"
                      f"{d:>9.3f}{d_fixed:>7.3f}{gain:>+5.0f}%")
        if ab_note:
            write_ab_note(out_dir, spec, spec_path, seams, seam_ts, overlap, fps)

    # Concatenate with crossfaded seams.
    bp = steps[0].bp
    go = steps[0].go
    tr = steps[0].tr
    for st in steps[1:]:
        b, g, t = crossfade_prev(bp, go, tr, st, overlap)
        bp = np.concatenate([bp[:-overlap], b, st.bp[overlap:]])
        go = np.concatenate([go[:-overlap], g, st.go[overlap:]])
        tr = np.concatenate([tr[:-overlap], t, st.tr[overlap:]])
    total = len(tr)
    print(f"[choreo] composed {total} frames = {total / fps:.1f}s")
    if dry_run:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    composed = {
        "smpl_params_global": {
            "body_pose": bp.astype(np.float32),
            "global_orient": go.astype(np.float32),
            "transl": tr.astype(np.float32),
            "betas": np.zeros((total, 10), dtype=np.float32),
        },
        "fps": fps,
        "num_frames": total,
        "mesh": steps[0].mesh,
        "source": {
            "dataset": "choreo-composer",
            "sequence": Path(spec_path).stem,
            "spec": spec,
            "note": "kinematic stitch of library segments; see choreo.py",
        },
    }
    pkl_path = out_dir / (Path(spec_path).stem + "_smpl_clean.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(composed, f)
    print(f"[choreo] wrote {pkl_path}")

    glb = export_mannequin(pkl_path)
    if no_viewer:
        return glb
    html = make_viewer(
        glb, scene=spec.get("scene"), title=spec.get("title"), fps=fps,
        out=out_dir / (Path(spec_path).stem + "_viewer.html"))
    return html


def crossfade_prev(bp, go, tr, nxt, overlap):
    """crossfade() against raw arrays for the accumulated sequence."""
    class _P:  # minimal shim
        pass
    p = _P()
    p.bp, p.go, p.tr = bp, go, tr
    return crossfade(p, nxt, overlap)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec", help="choreography spec JSON")
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the station table only (no export)")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--transitions", choices=("crossfade", "matched"), default=None,
                    help="seam mode (overrides the spec; default: spec value "
                         "or crossfade)")
    ap.add_argument("--ab-note", action="store_true",
                    help="matched mode: write <out>/transitions_ab.md with the "
                         "per-seam matched-vs-fixed evidence table")
    args = ap.parse_args()
    compose(args.spec, args.out_dir, args.dry_run, args.no_viewer, args.transitions,
            args.ab_note)


if __name__ == "__main__":
    sys.exit(main() or 0)
