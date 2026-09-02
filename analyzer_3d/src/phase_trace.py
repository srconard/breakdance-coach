"""Phase trace — per-frame kinematic descriptors for reading a move's phases.

CPU-only sibling of :mod:`contact_geometric`.  Where the contact stage answers
"which body part touches the floor when", this stage answers "where is the
body, which way is it turning, and how fast" so a breakdown's phases (over the
hands, kick, hand switch, front pass, ...) can be read off a cleaned SMPL
sequence instead of being hand-derived from ad-hoc scripts.  It reproduces
the method used for the first exemplar breakdown (flare, 2026-09-01) and
writes ``<stem>_phases.json`` next to the input pickle.

Input is a *cleaned* library pickle as written by :mod:`motion_cleanup` /
:mod:`nas_extract` (``smpl_params_global`` + ``fps`` + ``num_frames`` +
``mesh`` (+ optional ``keypoints_2d``)).  Y is up and the floor is y = 0.

What is MEASURED (forward kinematics over the saved pose, nothing fitted):

* heights above the floor and floor-plane (x, z) positions of head, pelvis,
  both wrists and both ankles per frame (wrist/ankle heights use the lower of
  joint and finger/toe tip, as the cleanup's contact detector does);
* two cumulative rotation signals: ``leg_azimuth_deg`` (heading of the
  ankle-midpoint around the pelvis in the floor plane — the exemplar's
  signal; held while the legs are stacked over the pelvis) and
  ``pelvis_yaw_deg`` (heading of the pelvis' hip axis — survives headspins
  and backspins where the leg azimuth is degenerate);
* rotation spans (sustained turning), 360-degree cycles inside them with
  per-cycle period, angular rate, body heights, hand-switch timing and floor
  footprint, plus the un-finished partial rotation at the end of each span;
* an ``inverted`` flag (head below pelvis), floor footprints (x/z extents of
  ankles, wrists, pelvis) and pelvis travel — ``travel_m`` is the path length
  of the pelvis after a 1 s smoothing (per-lap circling still leaks in on
  fast spins), ``net_displacement_m`` is start-to-end.

What is INFERRED (heuristic, labelled ``"inferred"`` in the output):

* ``valid_span`` — frames whose reconstruction is trusted.  Derived from the
  2D keypoint confidences saved with the pickle (close-ups and cuts lose the
  lower body) and a pelvis-height sanity range; when the pickle carries no
  keypoints every frame is treated as valid;
* cycle boundaries — anchored to head-height minima when a clear per-cycle
  head dip exists (``anchor: "head_min"``), otherwise at the 360-degree
  crossings of the rotation signal (``anchor: "azimuth_360"``);
* ``support_windows`` — while both feet are off the floor the body is
  assumed to rest on the lowest of wrists/head (``hands`` with which hand,
  ``head``), or to be ``airborne`` when nothing is low.  Only *ordering* of
  heights is observed, never contact — the geometric contact detector is
  blind to hand support in power moves, which is why this exists.

Examples::

    python -m analyzer_3d.src.phase_trace results/DI6kMAxMEch/DI6kMAxMEch_smpl_clean.pkl
    python -m analyzer_3d.src.phase_trace clip_smpl_clean.pkl --out output/ig_ingest/phases
    python -m analyzer_3d.src.phase_trace --batch /data/ig-extract/results --skip-existing
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from analyzer_3d.src.contact_geometric import (
    PKL_SUFFIX,
    _clip_arrays,
    _load_pkl,
    _stem_from_path,
    _true_runs,
    _write_json,
)
from analyzer_3d.src.contact_signatures import SMPL_JOINTS
from analyzer_3d.src.motion_cleanup import fk, savgol


METHOD = "phase_trace_v1"
JSON_SUFFIX = "_phases.json"

# Joints (SMPL indices) that the trace reports.  Effector heights take the
# lower of joint and tip, exactly like motion_cleanup.EFFECTORS.
J = {name: SMPL_JOINTS.index(name) for name in SMPL_JOINTS}
TRACKED = {
    "head": (J["head"], None),
    "pelvis": (J["pelvis"], None),
    "wrist_L": (J["L_wrist"], J["L_hand"]),
    "wrist_R": (J["R_wrist"], J["R_hand"]),
    "ankle_L": (J["L_ankle"], J["L_foot"]),
    "ankle_R": (J["R_ankle"], J["R_foot"]),
}

# Rotation detection
SMOOTH_S = 0.20             # Savitzky-Golay window for angles and heights
LEG_RADIUS_MIN_M = 0.15     # leg azimuth is held when the legs stack over the pelvis
HIP_AXIS_MAX_VERT = 0.95    # pelvis yaw is held when the hip axis is near vertical
RATE_MIN_DEG_S = 45.0       # sustained turning threshold for a rotation span
SPAN_GAP_S = 0.50           # dips below the rate threshold shorter than this join a span
SPAN_MIN_DEG = 300.0        # a span must turn at least this far to be reported
CYCLE_DEG = (270.0, 450.0)  # accepted azimuth travel of one 360-degree cycle
PERIOD_S = (0.35, 4.0)      # accepted cycle period
HEAD_ANCHOR_MAX_M = 0.62    # head must dip at least this low to anchor a cycle
HEAD_ANCHOR_DROP_M = 0.15   # ... and at least this far below its surroundings
ANCHOR_SNAP_FRAC = 0.40     # snap a 360-crossing to a head-min within this fraction of a period
HAND_SWITCH_DEBOUNCE_S = 0.20
TRAVEL_SMOOTH_S = 1.0       # pelvis smoothing before measuring floor travel

# Support inference
FEET_OFF_M = 0.12
FEET_OFF_MIN_S = 0.30
SUPPORT_AIRBORNE_M = 0.40   # nothing below this height -> airborne
SUPPORT_HEAD_MAX_M = 0.30   # head lowest and below this -> head support
BOTH_HANDS_M = 0.10         # wrists within this of each other -> both hands
SUPPORT_RUN_MIN_S = 0.15

# Valid-span inference (2D keypoint confidences saved by GVHMR, COCO-17 order)
KP_CONF_THRESH = 0.30
KP_VISIBLE_MIN = 10         # of 17 keypoints
KP_LOWER_BODY = slice(11, 17)
KP_LOWER_CONF_MIN = 0.15
KP_WINDOW_S = 0.50
PELVIS_RANGE_M = (0.0, 1.6)
INVALID_MIN_S = 0.30


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _load(path: Path) -> dict[str, Any]:
    """Load a cleaned pickle, silencing numpy's cross-version unpickle warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return _load_pkl(path)


def _smooth(series: np.ndarray, fps: float, window_s: float = SMOOTH_S, order: int = 3) -> np.ndarray:
    window = max(3, int(round(window_s * fps)) | 1)
    return savgol(series.astype(np.float64), window, polyorder=order, axis=0)


def _hold_where(values: np.ndarray, hold: np.ndarray) -> np.ndarray:
    """Forward-fill ``values`` over frames where ``hold`` is True."""
    out = values.astype(np.float64).copy()
    out[hold] = np.nan
    if np.all(np.isnan(out)):
        return np.zeros_like(out)
    first = int(np.flatnonzero(~np.isnan(out))[0])
    out[:first] = out[first]
    for f in range(first + 1, len(out)):
        if np.isnan(out[f]):
            out[f] = out[f - 1]
    return out


def _cumulative_deg(wrapped_rad: np.ndarray, hold: np.ndarray, fps: float) -> np.ndarray:
    held = _hold_where(wrapped_rad, hold)
    cumulative = np.degrees(np.unwrap(held))
    cumulative -= cumulative[0]
    return _smooth(cumulative, fps)


def _mode_filter(labels: np.ndarray, width: int) -> np.ndarray:
    """Sliding-window majority vote over small integer labels."""
    if width < 3 or len(labels) < width:
        return labels
    half = width // 2
    padded = np.pad(labels, half, mode="edge")
    out = labels.copy()
    n_labels = int(labels.max()) + 1
    for i in range(len(labels)):
        out[i] = int(np.bincount(padded[i:i + width], minlength=n_labels).argmax())
    return out


def _runs_min_length(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop True runs shorter than ``min_frames``."""
    out = mask.copy()
    for start, stop in _true_runs(mask):
        if stop - start < min_frames:
            out[start:stop] = False
    return out


def _merge_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, stop in runs:
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return [(s, e) for s, e in merged]


def _round(value: float, digits: int = 3) -> float:
    return float(round(float(value), digits))


def _bbox_xz(points: np.ndarray) -> dict[str, Any]:
    """Extent and centre of (N, 2) floor-plane points."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "extent_m": [_round(hi[0] - lo[0]), _round(hi[1] - lo[1])],
        "center": [_round(v) for v in (lo + hi) / 2.0],
    }


def _path_length(points_xz: np.ndarray) -> float:
    if len(points_xz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points_xz, axis=0), axis=1).sum())


def _travel(points_xz: np.ndarray, fps: float) -> float:
    """Floor travel: path length of the pelvis after a TRAVEL_SMOOTH_S smoothing,
    so per-cycle circling (a flare swings the hips ~0.3 m every lap) does not
    count as crossing the floor."""
    if len(points_xz) < 4:
        return _path_length(points_xz)
    return _path_length(_smooth(points_xz, fps, TRAVEL_SMOOTH_S, 2))


# ---------------------------------------------------------------------------
# measured series
# ---------------------------------------------------------------------------

def trace_kinematics(data: dict[str, Any], path: Path) -> dict[str, Any]:
    """Forward kinematics + rotation signals for every frame.  All measured."""
    body_pose, global_orient, transl, _, _, rest_joints, fps32 = _clip_arrays(data, path)
    fps = float(fps32)
    joints, rotations, _ = fk(
        body_pose.astype(np.float64),
        global_orient.astype(np.float64),
        transl.astype(np.float64),
        rest_joints.astype(np.float64),
    )
    frame_count = joints.shape[0]

    heights: dict[str, np.ndarray] = {}
    xz: dict[str, np.ndarray] = {}
    for name, (joint, tip) in TRACKED.items():
        y = joints[:, joint, 1]
        if tip is not None:
            y = np.minimum(y, joints[:, tip, 1])
        heights[name] = y
        xz[name] = joints[:, joint][:, [0, 2]]

    ankle_mid = (joints[:, J["L_ankle"]] + joints[:, J["R_ankle"]]) / 2.0
    leg_vec = ankle_mid[:, [0, 2]] - joints[:, J["pelvis"]][:, [0, 2]]
    leg_radius = np.linalg.norm(leg_vec, axis=1)
    leg_azimuth = _cumulative_deg(
        np.arctan2(leg_vec[:, 1], leg_vec[:, 0]), leg_radius < LEG_RADIUS_MIN_M, fps
    )

    hip_axis = rotations[:, J["pelvis"], :, 0]  # world direction of the pelvis' local +X
    pelvis_yaw = _cumulative_deg(
        np.arctan2(hip_axis[:, 2], hip_axis[:, 0]),
        np.abs(hip_axis[:, 1]) > HIP_AXIS_MAX_VERT,
        fps,
    )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "t": np.arange(frame_count) / fps,
        "heights": heights,
        "xz": xz,
        "head_smooth": _smooth(heights["head"], fps, 0.15, 2),
        "leg_azimuth_deg": leg_azimuth,
        "leg_radius_m": leg_radius,
        "pelvis_yaw_deg": pelvis_yaw,
        "inverted": heights["head"] < heights["pelvis"],
    }


# ---------------------------------------------------------------------------
# inferred: valid span
# ---------------------------------------------------------------------------

def infer_valid_span(data: dict[str, Any], kin: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Frames whose reconstruction is trusted.  Heuristic — see module docstring."""
    fps = kin["fps"]
    frame_count = kin["frame_count"]
    basis: list[str] = []
    valid = np.ones(frame_count, dtype=bool)

    pelvis = kin["heights"]["pelvis"]
    pelvis_ok = (pelvis >= PELVIS_RANGE_M[0]) & (pelvis <= PELVIS_RANGE_M[1])
    if not pelvis_ok.all():
        basis.append("pelvis_height_range")
    valid &= pelvis_ok

    keypoints = data.get("keypoints_2d")
    if keypoints is not None:
        keypoints = np.asarray(keypoints, dtype=np.float64)
        if keypoints.ndim == 3 and keypoints.shape[0] == frame_count and keypoints.shape[2] >= 3:
            conf = keypoints[:, :, 2]
            window = max(1, int(round(KP_WINDOW_S * fps)))
            kernel = np.ones(window) / window
            visible = np.convolve((conf > KP_CONF_THRESH).sum(axis=1), kernel, mode="same")
            lower = np.convolve(conf[:, KP_LOWER_BODY].mean(axis=1), kernel, mode="same")
            kp_ok = (visible >= KP_VISIBLE_MIN) & (lower >= KP_LOWER_CONF_MIN)
            valid &= kp_ok
            basis.append("keypoints_2d_confidence")
    else:
        basis.append("no_keypoints_2d")

    # Ignore invalid flickers shorter than INVALID_MIN_S
    invalid = _runs_min_length(~valid, max(1, int(round(INVALID_MIN_S * fps))))
    valid = ~invalid
    t = kin["t"]
    spans = [[_round(t[s], 2), _round(t[e - 1] + 1.0 / fps, 2)] for s, e in _true_runs(valid)]
    return valid, {
        "label": "inferred",
        "basis": basis,
        "valid_frac": _round(valid.mean()),
        "spans_s": spans,
    }


# ---------------------------------------------------------------------------
# rotation spans and cycles
# ---------------------------------------------------------------------------

def _rotation_spans(cumulative: np.ndarray, fps: float) -> list[tuple[int, int]]:
    rate = np.gradient(cumulative) * fps
    fast = np.abs(rate) >= RATE_MIN_DEG_S
    runs = _merge_runs(_true_runs(fast), int(round(SPAN_GAP_S * fps)))
    return [
        (s, e) for s, e in runs
        if abs(cumulative[e - 1] - cumulative[s]) >= SPAN_MIN_DEG
    ]


def _head_anchors(head: np.ndarray, fps: float, period_frames: int) -> np.ndarray:
    """Local minima of the head height that dip clearly below their surroundings."""
    n = len(head)
    if n < 3:
        return np.zeros(0, dtype=int)
    interior = head[1:-1]
    local_min = (interior <= head[:-2]) & (interior < head[2:]) & (interior < HEAD_ANCHOR_MAX_M)
    candidates = np.flatnonzero(local_min) + 1
    half = max(2, period_frames // 2)
    keep = []
    for f in candidates:
        lo, hi = max(0, f - half), min(n, f + half + 1)
        if head[lo:hi].max() - head[f] >= HEAD_ANCHOR_DROP_M:
            keep.append(int(f))
    # collapse minima closer than a quarter period (keep the lowest)
    anchors: list[int] = []
    for f in keep:
        if anchors and f - anchors[-1] < max(2, period_frames // 4):
            if head[f] < head[anchors[-1]]:
                anchors[-1] = f
        else:
            anchors.append(f)
    return np.asarray(anchors, dtype=int)


def _crossings(cumulative: np.ndarray, start: int, stop: int) -> list[int]:
    """Frames inside [start, stop) where the signal completes each further 360 degrees."""
    origin = cumulative[start]
    out = [start]
    k = 1
    sign = 1.0 if cumulative[stop - 1] >= origin else -1.0
    for f in range(start + 1, stop):
        while sign * (cumulative[f] - origin) >= 360.0 * k:
            out.append(f)
            k += 1
    return out


def _hand_switch_fracs(kin: dict[str, Any], start: int, stop: int) -> list[float]:
    fps = kin["fps"]
    lower_is_left = kin["heights"]["wrist_L"][start:stop] < kin["heights"]["wrist_R"][start:stop]
    flips = np.flatnonzero(np.diff(lower_is_left.astype(np.int8)) != 0) + 1
    debounce = int(round(HAND_SWITCH_DEBOUNCE_S * fps))
    events: list[int] = []
    for f in flips:
        if not events or f - events[-1] >= debounce:
            events.append(int(f))
    length = max(1, stop - start)
    return [_round(f / length, 2) for f in events]


def _cycle_record(kin: dict[str, Any], cumulative: np.ndarray, start: int, stop: int,
                  index: int, anchors: tuple[str, str], valid: np.ndarray) -> dict[str, Any]:
    fps = kin["fps"]
    t = kin["t"]
    h = kin["heights"]
    xz = kin["xz"]
    rate = np.gradient(cumulative[start:stop + 1]) * fps if stop + 1 <= len(cumulative) else np.gradient(cumulative[start:stop]) * fps
    sl = slice(start, stop)
    ankles = np.concatenate([xz["ankle_L"][sl], xz["ankle_R"][sl]])
    wrists = np.concatenate([xz["wrist_L"][sl], xz["wrist_R"][sl]])
    return {
        "index": index,
        "start_s": _round(t[start]),
        "end_s": _round(t[stop]) if stop < len(t) else _round(t[-1] + 1.0 / fps),
        "period_s": _round((stop - start) / fps),
        "azimuth_delta_deg": _round(cumulative[stop] - cumulative[start], 1) if stop < len(cumulative) else _round(cumulative[-1] - cumulative[start], 1),
        "anchor": {"start": anchors[0], "end": anchors[1]},
        "valid": bool(valid[sl].mean() > 0.5),
        "rate_deg_s": {"mean": _round(rate.mean(), 1), "peak": _round(rate[np.abs(rate).argmax()], 1)},
        "heights_m": {
            "head": [_round(h["head"][sl].min()), _round(h["head"][sl].max())],
            "pelvis": [_round(h["pelvis"][sl].min()), _round(h["pelvis"][sl].max())],
            "ankle_L_peak": _round(h["ankle_L"][sl].max()),
            "ankle_R_peak": _round(h["ankle_R"][sl].max()),
            "wrist_L_min": _round(h["wrist_L"][sl].min()),
            "wrist_R_min": _round(h["wrist_R"][sl].min()),
        },
        "hand_switch_fracs": _hand_switch_fracs(kin, start, stop),
        "inverted_frac": _round(kin["inverted"][sl].mean(), 2),
        "footprint": {"ankles": _bbox_xz(ankles), "wrists": _bbox_xz(wrists)},
        "pelvis_travel_m": _round(_travel(xz["pelvis"][sl], fps)),
        "pelvis_net_m": _round(float(np.linalg.norm(xz["pelvis"][stop - 1] - xz["pelvis"][start]))),
    }


def detect_cycles(kin: dict[str, Any], cumulative: np.ndarray, valid: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rotation spans and the 360-degree cycles inside them for one signal."""
    fps = kin["fps"]
    t = kin["t"]
    head = kin["head_smooth"]
    spans_out: list[dict[str, Any]] = []
    cycles: list[dict[str, Any]] = []

    for span_start, span_stop in _rotation_spans(cumulative, fps):
        total = cumulative[span_stop - 1] - cumulative[span_start]
        n_full = int(abs(total) // 360.0)
        period_frames = int(round((span_stop - span_start) / max(1, n_full))) if n_full else span_stop - span_start
        crossings = _crossings(cumulative, span_start, span_stop)
        anchors = _head_anchors(head[span_start:span_stop], fps, period_frames) + span_start
        snap = int(round(ANCHOR_SNAP_FRAC * period_frames))

        # The span end closes a final cycle when the remainder is itself a
        # near-full rotation (a single 360 drill that stops just short).
        if abs(cumulative[span_stop - 1] - cumulative[crossings[-1]]) >= CYCLE_DEG[0]:
            crossings.append(span_stop - 1)

        min_sep = max(2, int(round(PERIOD_S[0] * fps)))

        def bounds_for(use_anchors: bool) -> list[tuple[int, str]]:
            bounds: list[tuple[int, str]] = []
            for f in crossings:
                kind = "azimuth_360" if f != span_stop - 1 else "span_end"
                if use_anchors and len(anchors):
                    nearest = int(anchors[np.abs(anchors - f).argmin()])
                    if abs(nearest - f) <= snap:
                        f, kind = nearest, "head_min"
                if not bounds or f - bounds[-1][0] >= min_sep:
                    bounds.append((f, kind))
            return bounds

        def cycles_for(bounds: list[tuple[int, str]]) -> list[tuple[int, int, tuple[str, str]]]:
            accepted = []
            for (s, kind_s), (e, kind_e) in zip(bounds, bounds[1:]):
                delta = abs(cumulative[e] - cumulative[s])
                period = (e - s) / fps
                if CYCLE_DEG[0] <= delta <= CYCLE_DEG[1] and PERIOD_S[0] <= period <= PERIOD_S[1]:
                    accepted.append((s, e, (kind_s, kind_e)))
            return accepted

        # Head-minimum anchoring is physically meaningful (the over-the-hands
        # moment of a flare) but only when the head dips once per lap; on a
        # backspin or 1990 the head sits low throughout and its noise minima
        # would swallow cycles.  Use the anchors only when they lose nothing.
        anchored = cycles_for(bounds_for(True))
        raw = cycles_for(bounds_for(False))
        chosen = anchored if len(anchored) >= len(raw) else raw
        bounds = bounds_for(len(anchored) >= len(raw))

        span_cycles: list[dict[str, Any]] = []
        for s, e, kinds in chosen:
            span_cycles.append(_cycle_record(kin, cumulative, s, e, len(cycles) + len(span_cycles), kinds, valid))

        last = bounds[-1][0] if bounds else span_start
        partial = cumulative[span_stop - 1] - cumulative[last]
        periods = [c["period_s"] for c in span_cycles]
        pelvis_xz = kin["xz"]["pelvis"][span_start:span_stop]
        ankles = np.concatenate([kin["xz"]["ankle_L"][span_start:span_stop], kin["xz"]["ankle_R"][span_start:span_stop]])
        mean_pelvis = pelvis_xz.mean(axis=0)
        spans_out.append({
            "start_s": _round(t[span_start]),
            "end_s": _round(t[span_stop - 1] + 1.0 / fps),
            "duration_s": _round((span_stop - span_start) / fps),
            "total_deg": _round(total, 1),
            "direction": "+" if total >= 0 else "-",
            "n_cycles": len(span_cycles),
            "cycle_indices": [c["index"] for c in span_cycles],
            "partial_deg": _round(partial, 1),
            "mean_period_s": _round(float(np.mean(periods))) if periods else None,
            "sd_period_s": _round(float(np.std(periods))) if len(periods) > 1 else None,
            "valid": bool(valid[span_start:span_stop].mean() > 0.5),
            "pelvis_travel_m": _round(_travel(pelvis_xz, fps)),
            "pelvis_net_m": _round(float(np.linalg.norm(pelvis_xz[-1] - pelvis_xz[0]))),
            "footprint": {"ankles": _bbox_xz(ankles)},
            "max_ankle_radius_m": _round(float(np.linalg.norm(ankles - mean_pelvis, axis=1).max())),
        })
        cycles.extend(span_cycles)
    return spans_out, cycles


# ---------------------------------------------------------------------------
# inferred: support windows
# ---------------------------------------------------------------------------

SUPPORT_LABELS = ["hands_L", "hands_R", "hands_both", "head", "airborne"]


def infer_support_windows(kin: dict[str, Any], cumulative: np.ndarray, valid: np.ndarray) -> list[dict[str, Any]]:
    """Support while both feet are off the floor.  Inferred from height ordering only."""
    fps = kin["fps"]
    t = kin["t"]
    h = kin["heights"]
    feet_off = (h["ankle_L"] > FEET_OFF_M) & (h["ankle_R"] > FEET_OFF_M)
    feet_off = _mode_filter(feet_off.astype(np.int64), max(3, int(round(0.10 * fps)) | 1)).astype(bool)
    feet_off = _runs_min_length(feet_off, int(round(FEET_OFF_MIN_S * fps)))

    wl, wr, head = h["wrist_L"], h["wrist_R"], h["head"]
    lowest = np.minimum(np.minimum(wl, wr), head)
    labels = np.full(len(t), SUPPORT_LABELS.index("hands_both"), dtype=np.int64)
    labels[wl < wr - BOTH_HANDS_M] = SUPPORT_LABELS.index("hands_L")
    labels[wr < wl - BOTH_HANDS_M] = SUPPORT_LABELS.index("hands_R")
    labels[(head <= np.minimum(wl, wr)) & (head < SUPPORT_HEAD_MAX_M)] = SUPPORT_LABELS.index("head")
    labels[lowest > SUPPORT_AIRBORNE_M] = SUPPORT_LABELS.index("airborne")
    labels = _mode_filter(labels, max(3, int(round(SUPPORT_RUN_MIN_S * fps)) | 1))

    windows: list[dict[str, Any]] = []
    min_run = max(1, int(round(SUPPORT_RUN_MIN_S * fps)))
    for span_start, span_stop in _true_runs(feet_off):
        runs: list[list[int]] = []
        for f in range(span_start, span_stop):
            if runs and runs[-1][2] == labels[f]:
                runs[-1][1] = f + 1
            else:
                runs.append([f, f + 1, int(labels[f])])
        # absorb runs shorter than the minimum into their predecessor
        merged: list[list[int]] = []
        for run in runs:
            if merged and run[1] - run[0] < min_run:
                merged[-1][1] = run[1]
            else:
                merged.append(run)
        for s, e, label in merged:
            name = SUPPORT_LABELS[label]
            sl = slice(s, e)
            windows.append({
                "start_s": _round(t[s]),
                "end_s": _round(t[e - 1] + 1.0 / fps),
                "duration_s": _round((e - s) / fps),
                "support": "hands" if name.startswith("hands") else name,
                "hand": name.split("_", 1)[1] if name.startswith("hands") else None,
                "label": "inferred",
                "basis": "feet_off_floor + lowest of wrists/head",
                "valid": bool(valid[sl].mean() > 0.5),
                "wrist_min_m": [_round(wl[sl].min()), _round(wr[sl].min())],
                "wrist_mean_m": [_round(wl[sl].mean()), _round(wr[sl].mean())],
                "head_min_m": _round(head[sl].min()),
                "pelvis_mean_m": _round(h["pelvis"][sl].mean()),
                "azimuth_delta_deg": _round(cumulative[e - 1] - cumulative[s], 1),
                "n_hand_switches": len(_hand_switch_fracs(kin, s, e)),
            })
    return windows


# ---------------------------------------------------------------------------
# document assembly
# ---------------------------------------------------------------------------

def _series_block(kin: dict[str, Any], valid: np.ndarray, rate: np.ndarray, stride: int) -> dict[str, Any]:
    idx = np.arange(0, kin["frame_count"], stride)
    r3 = lambda a: [_round(v) for v in a[idx]]  # noqa: E731
    r1 = lambda a: [_round(v, 1) for v in a[idx]]  # noqa: E731
    return {
        "stride_frames": stride,
        "frame": [int(i) for i in idx],
        "t": r3(kin["t"]),
        "valid": [bool(v) for v in valid[idx]],
        "inverted": [bool(v) for v in kin["inverted"][idx]],
        "heights_m": {name: r3(series) for name, series in kin["heights"].items()},
        "xz_m": {name: [[_round(p[0]), _round(p[1])] for p in series[idx]] for name, series in kin["xz"].items()},
        "leg_azimuth_deg": r1(kin["leg_azimuth_deg"]),
        "pelvis_yaw_deg": r1(kin["pelvis_yaw_deg"]),
        "rate_deg_s": r1(rate),
    }


def _params_block() -> dict[str, Any]:
    return {
        "smooth_s": SMOOTH_S,
        "leg_radius_min_m": LEG_RADIUS_MIN_M,
        "rate_min_deg_s": RATE_MIN_DEG_S,
        "span_gap_s": SPAN_GAP_S,
        "span_min_deg": SPAN_MIN_DEG,
        "cycle_deg": list(CYCLE_DEG),
        "period_s": list(PERIOD_S),
        "head_anchor_max_m": HEAD_ANCHOR_MAX_M,
        "head_anchor_drop_m": HEAD_ANCHOR_DROP_M,
        "anchor_snap_frac": ANCHOR_SNAP_FRAC,
        "hand_switch_debounce_s": HAND_SWITCH_DEBOUNCE_S,
        "travel_smooth_s": TRAVEL_SMOOTH_S,
        "feet_off_m": FEET_OFF_M,
        "feet_off_min_s": FEET_OFF_MIN_S,
        "support_airborne_m": SUPPORT_AIRBORNE_M,
        "support_head_max_m": SUPPORT_HEAD_MAX_M,
        "both_hands_m": BOTH_HANDS_M,
        "kp_conf_thresh": KP_CONF_THRESH,
        "kp_visible_min": KP_VISIBLE_MIN,
        "kp_lower_conf_min": KP_LOWER_CONF_MIN,
        "pelvis_range_m": list(PELVIS_RANGE_M),
    }


def build_phase_trace(path: str | Path, stride: int = 1) -> dict[str, Any]:
    """Compute the full phase-trace document for one cleaned pickle."""
    path = Path(path)
    data = _load(path)
    kin = trace_kinematics(data, path)
    fps = kin["fps"]
    valid, valid_info = infer_valid_span(data, kin)

    candidates = {}
    for name in ("leg_azimuth", "pelvis_yaw"):
        spans, cycles = detect_cycles(kin, kin[f"{name}_deg"], valid)
        candidates[name] = (spans, cycles)
    # Prefer the signal that resolves more cycles; the exemplar's leg azimuth wins ties.
    signal = max(("leg_azimuth", "pelvis_yaw"), key=lambda n: (len(candidates[n][1]), n == "leg_azimuth"))
    spans, cycles = candidates[signal]
    cumulative = kin[f"{signal}_deg"]
    rate = np.gradient(cumulative) * fps
    support = infer_support_windows(kin, cumulative, valid)

    pelvis_xz = kin["xz"]["pelvis"]
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) == 0:
        valid_idx = np.arange(kin["frame_count"])
    pelvis_valid = pelvis_xz[valid_idx]
    ankles_valid = np.concatenate([kin["xz"]["ankle_L"][valid_idx], kin["xz"]["ankle_R"][valid_idx]])
    wrists_valid = np.concatenate([kin["xz"]["wrist_L"][valid_idx], kin["xz"]["wrist_R"][valid_idx]])
    periods = [c["period_s"] for c in cycles]
    hand_support_s = sum(w["duration_s"] for w in support if w["support"] == "hands")

    summary = {
        "n_cycles": len(cycles),
        "n_rotation_spans": len(spans),
        "rotation_signal": signal,
        "mean_period_s": _round(float(np.mean(periods))) if periods else None,
        "sd_period_s": _round(float(np.std(periods))) if len(periods) > 1 else None,
        "total_rotation_deg": _round(sum(abs(s["total_deg"]) for s in spans), 1),
        "leg_azimuth_net_deg": _round(kin["leg_azimuth_deg"][-1] - kin["leg_azimuth_deg"][0], 1),
        "pelvis_yaw_net_deg": _round(kin["pelvis_yaw_deg"][-1] - kin["pelvis_yaw_deg"][0], 1),
        "travel_m": _round(_travel(pelvis_valid, fps)),
        "net_displacement_m": _round(float(np.linalg.norm(pelvis_valid[-1] - pelvis_valid[0]))),
        "footprint": {
            "pelvis": _bbox_xz(pelvis_valid),
            "ankles": _bbox_xz(ankles_valid),
            "wrists": _bbox_xz(wrists_valid),
        },
        "n_support_windows": len(support),
        "hand_support_s": _round(hand_support_s, 2),
        "inverted_frac": _round(kin["inverted"][valid_idx].mean(), 3),
        "valid_frac": valid_info["valid_frac"],
        "heights_m": {
            name: [_round(series[valid_idx].min()), _round(series[valid_idx].max())]
            for name, series in kin["heights"].items()
        },
    }

    return {
        "method": METHOD,
        "clip": _stem_from_path(path),
        "source_pkl": path.name,
        "fps": _round(fps, 4),
        "num_frames": kin["frame_count"],
        "duration_s": _round(kin["frame_count"] / fps),
        "floor_y": 0.0 if data.get("cleanup", {}).get("rebased_to_zero", True) else data["cleanup"].get("floor_y_original"),
        "params": _params_block(),
        "summary": summary,
        "valid_span": valid_info,
        "rotation_spans": spans,
        "cycles": cycles,
        "support_windows": support,
        "series": _series_block(kin, valid, rate, stride),
    }


def _one_line(document: dict[str, Any]) -> str:
    s = document["summary"]
    period = f"{s['mean_period_s']:.2f}s" if s["mean_period_s"] else "-"
    return (
        f"{document['clip']}: {document['num_frames']}f {document['duration_s']:.1f}s "
        f"cycles={s['n_cycles']} ({s['rotation_signal']}) period={period} "
        f"spans={s['n_rotation_spans']} support={s['n_support_windows']} "
        f"hands={s['hand_support_s']:.1f}s travel={s['travel_m']:.2f}m valid={s['valid_frac']:.2f}"
    )


def trace_pkl(path: str | Path, out_dir: str | Path | None = None, stride: int = 1) -> Path:
    """Trace one cleaned pickle and write ``<stem>_phases.json``.  Returns the path."""
    path = Path(path)
    document = build_phase_trace(path, stride=stride)
    target_dir = Path(out_dir) if out_dir is not None else path.parent
    target = target_dir / f"{document['clip']}{JSON_SUFFIX}"
    _write_json(target, document)
    print(_one_line(document), flush=True)
    return target


def _discover(results_dir: Path) -> list[Path]:
    direct = sorted(results_dir.glob(f"*{PKL_SUFFIX}"))
    nested = sorted(p for p in results_dir.glob(f"*/*{PKL_SUFFIX}"))
    return direct + nested


def run_batch(results_dir: Path, out_dir: Path | None, stride: int, skip_existing: bool) -> int:
    """Trace every cleaned pickle under ``results_dir``.  Returns the failure count."""
    pkls = _discover(results_dir)
    if not pkls:
        print(f"no {PKL_SUFFIX} files under {results_dir}", file=sys.stderr)
        return 1
    traced = skipped = failed = 0
    for pkl in pkls:
        stem = _stem_from_path(pkl)
        target = (out_dir if out_dir is not None else pkl.parent) / f"{stem}{JSON_SUFFIX}"
        if skip_existing and target.is_file():
            skipped += 1
            continue
        try:
            trace_pkl(pkl, out_dir, stride)
            traced += 1
        except Exception as exc:  # keep the batch going, report at the end
            failed += 1
            print(f"{stem}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print(f"phase_trace: traced={traced} skipped={skipped} failed={failed} of {len(pkls)}", flush=True)
    return failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-frame kinematic phase trace over a cleaned SMPL pickle"
    )
    parser.add_argument("pkl", nargs="?", type=Path, help=f"a *{PKL_SUFFIX} file")
    parser.add_argument("--batch", type=Path, default=None,
                        help="results directory: trace every *_smpl_clean.pkl in it or one level down")
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for the JSON output (default: next to each pickle)")
    parser.add_argument("--stride", type=_positive_int, default=1,
                        help="keep every Nth frame in the per-frame series (analysis always runs at full rate)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="batch mode: skip clips whose phases JSON already exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch is None and args.pkl is None:
        build_parser().error("give a pickle or --batch DIR")
    if args.batch is not None:
        return 1 if run_batch(args.batch, args.out, args.stride, args.skip_existing) else 0
    trace_pkl(args.pkl, args.out, args.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
