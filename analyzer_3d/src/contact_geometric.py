"""Geometric mesh-vs-floor contact signatures for the reference library.

This is the CPU-only replacement for the failed DECO learned-prior route.
It poses each cleaned SMPL sequence with numpy linear blend skinning, then
marks a named body part as contacting the floor when its lowest vertices are
both near the floor and slow in world XZ.  The result is a compact set of
time-windowed contacts for the reference-library affordance compiler.

This module complements :mod:`contact_refine`, which optimizes an individual
video reconstruction around supplied contact intervals; it does not optimize
or modify motion.  It uses only Python's standard library and numpy.

Examples::

    python -m analyzer_3d.src.contact_geometric stats clip_smpl_clean.pkl
    python -m analyzer_3d.src.contact_geometric build clip_smpl_clean.pkl
    python -m analyzer_3d.src.contact_geometric library
    python -m analyzer_3d.src.contact_geometric inspect CLIP_STEM --part palm_L
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyzer_3d.src.contact_signatures import (
    PART_GROUPS,
    SMPL_JOINTS,
    build_vertex_parts,
)
from analyzer_3d.src.motion_cleanup import PARENTS24, fk


# Detection thresholds.  These are deliberately module-level and explicit so
# tuning a body part does not require changing the detector itself.
H_ENTER = {
    "palm_L": 0.05,
    "palm_R": 0.05,
    "forearm_L": 0.06,
    "forearm_R": 0.06,
    "shoulder_L": 0.08,
    "shoulder_R": 0.08,
    "foot_L": 0.05,
    "foot_R": 0.05,
    "knee_L": 0.06,
    "knee_R": 0.06,
    "head": 0.08,
    "back_glutes": 0.08,
}

V_MAX = {
    "palm_L": 0.60,
    "palm_R": 0.60,
    "forearm_L": 0.60,
    "forearm_R": 0.60,
    "shoulder_L": 0.90,
    "shoulder_R": 0.90,
    "foot_L": 0.60,
    "foot_R": 0.60,
    "knee_L": 0.60,
    "knee_R": 0.60,
    "head": 0.90,
    "back_glutes": 0.90,
}

H_EXIT_EXTRA_M = 0.02
BAND_M = 0.03
MEDIAN_WIN_S = 0.10
MERGE_GAP_S = 0.25
MIN_WINDOW_S = {
    "default": 0.15,
    "back_glutes": 0.25,
    "shoulder_L": 0.25,
    "shoulder_R": 0.25,
    "head": 0.25,
}

FLOOR_WARNING_M = 0.015
FRAME_CHUNK = 256
PKL_SUFFIX = "_smpl_clean.pkl"
DEFAULT_ROOT = Path("output/reference_library")
DEFAULT_OUT = DEFAULT_ROOT / "contact_signatures.json"
DEFAULT_VALIDATION_OUT = DEFAULT_ROOT / "contact_geometric_validation.txt"

EFFECTOR_PARTS = {
    "L_ankle": ("foot_L",),
    "R_ankle": ("foot_R",),
    "L_wrist": ("palm_L", "forearm_L"),
    "R_wrist": ("palm_R", "forearm_R"),
}


@dataclass(frozen=True)
class GeometrySeries:
    """Per-frame measurements used by detection and debugging."""

    heights: np.ndarray
    speeds: np.ndarray
    global_min_heights: np.ndarray
    floor_offset_m: float
    floor_y_used: float


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _load_pkl(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: pickle root must be a dict")
    for key in ("smpl_params_global", "mesh", "fps", "num_frames"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")
    return data


def _clip_arrays(data: dict[str, Any], path: str | Path) -> tuple[np.ndarray, ...]:
    """Validate a cleaned pickle and return float32 arrays plus fps."""
    params = data["smpl_params_global"]
    mesh = data["mesh"]
    for key in ("body_pose", "global_orient", "transl"):
        if key not in params:
            raise ValueError(f"{path}: smpl_params_global missing {key!r}")
    for key in ("vertices", "weights", "joints"):
        if key not in mesh:
            raise ValueError(f"{path}: mesh missing {key!r}")

    body_pose = np.asarray(params["body_pose"], dtype=np.float32)
    global_orient = np.asarray(params["global_orient"], dtype=np.float32)
    transl = np.asarray(params["transl"], dtype=np.float32)
    rest_vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    weights = np.asarray(mesh["weights"], dtype=np.float32)
    rest_joints = np.asarray(mesh["joints"], dtype=np.float32)
    fps = float(data["fps"])

    frame_count = body_pose.shape[0]
    if body_pose.shape != (frame_count, 63):
        raise ValueError(f"{path}: body_pose must have shape (T, 63)")
    if global_orient.shape != (frame_count, 3):
        raise ValueError(f"{path}: global_orient must have shape (T, 3)")
    if transl.shape != (frame_count, 3):
        raise ValueError(f"{path}: transl must have shape (T, 3)")
    if frame_count < 1:
        raise ValueError(f"{path}: clip has no frames")
    if int(data["num_frames"]) != frame_count:
        raise ValueError(
            f"{path}: num_frames={data['num_frames']} but arrays contain {frame_count}"
        )
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError(f"{path}: fps must be finite and positive")
    if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
        raise ValueError(f"{path}: mesh vertices must have shape (V, 3)")
    if weights.shape != (rest_vertices.shape[0], len(SMPL_JOINTS)):
        raise ValueError(f"{path}: mesh weights must have shape (V, 24)")
    if rest_joints.shape != (len(PARENTS24), 3):
        raise ValueError(f"{path}: mesh joints must have shape (24, 3)")
    if len(SMPL_JOINTS) != len(PARENTS24):
        raise ValueError("SMPL joint names and FK parent table disagree")

    return (
        body_pose,
        global_orient,
        transl,
        rest_vertices,
        weights,
        rest_joints,
        np.float32(fps),
    )


def _pose_vertices(
    global_rotations: np.ndarray,
    joint_positions: np.ndarray,
    transl: np.ndarray,
    rest_vertices: np.ndarray,
    weights: np.ndarray,
    rest_joints: np.ndarray,
) -> np.ndarray:
    """Pose a frame chunk at full XYZ using float32 linear blend skinning."""
    global_rotations = np.asarray(global_rotations, dtype=np.float32)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    transl = np.asarray(transl, dtype=np.float32)
    joint_translation = joint_positions - np.einsum(
        "fjab,jb->fja", global_rotations, rest_joints, optimize=True
    )

    frame_count = global_rotations.shape[0]
    vertex_count = rest_vertices.shape[0]
    posed = np.empty((frame_count, vertex_count, 3), dtype=np.float32)
    for axis in range(3):
        blended_row = np.einsum(
            "vj,fjb->fvb",
            weights,
            global_rotations[:, :, axis, :],
            optimize=True,
        )
        rotated = np.einsum(
            "fvb,vb->fv", blended_row, rest_vertices, optimize=True
        )
        translated = np.einsum(
            "vj,fj->fv", weights, joint_translation[:, :, axis], optimize=True
        )
        posed[:, :, axis] = rotated + translated + transl[:, axis, None]
    return posed


def _pose_vertex_y(
    global_rotations: np.ndarray,
    joint_positions: np.ndarray,
    transl: np.ndarray,
    rest_vertices: np.ndarray,
    weights: np.ndarray,
    rest_joints: np.ndarray,
) -> np.ndarray:
    """Pose only Y for all vertices when detection itself is strided."""
    global_rotations = np.asarray(global_rotations, dtype=np.float32)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    transl = np.asarray(transl, dtype=np.float32)
    joint_translation_y = joint_positions[:, :, 1] - np.einsum(
        "fjb,jb->fj", global_rotations[:, :, 1, :], rest_joints, optimize=True
    )
    blended_row = np.einsum(
        "vj,fjb->fvb",
        weights,
        global_rotations[:, :, 1, :],
        optimize=True,
    )
    rotated = np.einsum(
        "fvb,vb->fv", blended_row, rest_vertices, optimize=True
    )
    translated = np.einsum(
        "vj,fj->fv", weights, joint_translation_y, optimize=True
    )
    return rotated + translated + transl[:, 1, None]


def compute_geometry(
    data: dict[str, Any],
    vertex_parts: np.ndarray,
    part_names: list[str],
    *,
    stride: int = 1,
    chunk_size: int = FRAME_CHUNK,
    label: str = "clip",
) -> GeometrySeries:
    """Compute per-frame part height and material-XZ-speed measurements.

    The XYZ LBS pass is frame-chunked and optionally vertex-strided.  Floor
    verification always uses all mesh vertices, including when ``stride`` is
    greater than one.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    (
        body_pose,
        global_orient,
        transl,
        rest_vertices,
        weights,
        rest_joints,
        fps_value,
    ) = _clip_arrays(data, label)
    fps = float(fps_value)
    frame_count = body_pose.shape[0]
    vertex_count = rest_vertices.shape[0]

    vertex_parts = np.asarray(vertex_parts, dtype=np.int64)
    if vertex_parts.shape != (vertex_count,):
        raise ValueError(
            f"{label}: vertex partition has {vertex_parts.shape}, expected {(vertex_count,)}"
        )
    if part_names != list(PART_GROUPS.keys()):
        raise ValueError("part order from build_vertex_parts does not match PART_GROUPS")

    selected = np.arange(0, vertex_count, stride, dtype=np.int64)
    selected_parts = vertex_parts[selected]
    for part_index, part_name in enumerate(part_names):
        if not np.any(selected_parts == part_index):
            raise ValueError(
                f"stride {stride} leaves part {part_name!r} with no vertices"
            )

    # FK is small compared with the mesh and is retained for all frames.  The
    # returned joint positions are unused; G and pos drive the chunked LBS.
    _, global_rotations, joint_positions = fk(
        body_pose, global_orient, transl, rest_joints
    )
    global_rotations = np.asarray(global_rotations, dtype=np.float32)
    joint_positions = np.asarray(joint_positions, dtype=np.float32)

    selected_vertices = rest_vertices[selected]
    selected_weights = weights[selected]
    part_count = len(part_names)
    absolute_heights = np.empty((frame_count, part_count), dtype=np.float32)
    speeds = np.empty((frame_count, part_count), dtype=np.float32)
    global_min = np.empty(frame_count, dtype=np.float32)
    dt = 1.0 / fps

    for start in range(0, frame_count, chunk_size):
        stop = min(frame_count, start + chunk_size)
        halo_start = max(0, start - 1)
        halo_stop = min(frame_count, stop + 1)
        posed = _pose_vertices(
            global_rotations[halo_start:halo_stop],
            joint_positions[halo_start:halo_stop],
            transl[halo_start:halo_stop],
            selected_vertices,
            selected_weights,
            rest_joints,
        )
        core = slice(start - halo_start, stop - halo_start)

        if stride == 1:
            global_min[start:stop] = posed[core, :, 1].min(axis=1)
        else:
            full_y = _pose_vertex_y(
                global_rotations[start:stop],
                joint_positions[start:stop],
                transl[start:stop],
                rest_vertices,
                weights,
                rest_joints,
            )
            global_min[start:stop] = full_y.min(axis=1)

        frame_indices = np.arange(start, stop, dtype=np.int64)
        if frame_count == 1:
            vertex_speeds = np.zeros((1, len(selected)), dtype=np.float32)
        else:
            previous = np.maximum(frame_indices - 1, 0)
            following = np.minimum(frame_indices + 1, frame_count - 1)
            previous_local = previous - halo_start
            following_local = following - halo_start
            delta_xz = (
                posed[following_local][:, :, (0, 2)]
                - posed[previous_local][:, :, (0, 2)]
            )
            elapsed = ((following - previous) * dt).astype(np.float32)
            vertex_speeds = np.linalg.norm(delta_xz, axis=2) / elapsed[:, None]

        current_y = posed[core, :, 1]
        for part_index in range(part_count):
            part_vertices = selected_parts == part_index
            part_y = current_y[:, part_vertices]
            part_min = part_y.min(axis=1)
            absolute_heights[start:stop, part_index] = part_min
            in_band = part_y < (part_min[:, None] + BAND_M)
            part_speed = vertex_speeds[:, part_vertices]
            speeds[start:stop, part_index] = (
                np.where(in_band, part_speed, 0.0).sum(axis=1)
                / in_band.sum(axis=1)
            )

    if not (
        np.isfinite(absolute_heights).all()
        and np.isfinite(speeds).all()
        and np.isfinite(global_min).all()
    ):
        raise ValueError(f"{label}: non-finite values produced by LBS")

    measured_floor = max(
        float(np.percentile(global_min, 5.0)), float(global_min.min())
    )
    if abs(measured_floor) > FLOOR_WARNING_M:
        print(
            f"WARNING: {label}: measured floor y={measured_floor:+.4f} m; "
            "using it instead of y=0",
            file=sys.stderr,
        )
        floor_y_used = measured_floor
    else:
        floor_y_used = 0.0

    return GeometrySeries(
        heights=(absolute_heights - np.float32(floor_y_used)).astype(
            np.float32, copy=False
        ),
        speeds=speeds,
        global_min_heights=global_min,
        floor_offset_m=measured_floor,
        floor_y_used=floor_y_used,
    )


def _median_window_frames(fps: float) -> int:
    width = max(1, int(round(MEDIAN_WIN_S * fps)))
    if width % 2 == 0:
        width += 1
    return width


def _median_filter_bool(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return mask.copy()
    half = width // 2
    padded = np.pad(mask.astype(np.int16), (half, half), mode="edge")
    votes = np.convolve(padded, np.ones(width, dtype=np.int16), mode="valid")
    return votes >= (half + 1)


def _hysteresis_mask(
    heights: np.ndarray,
    speeds: np.ndarray,
    part_name: str,
) -> np.ndarray:
    enter_h = H_ENTER[part_name]
    exit_h = enter_h + H_EXIT_EXTRA_M
    enter_v = V_MAX[part_name]
    exit_v = enter_v * 1.5
    result = np.zeros(len(heights), dtype=bool)
    contacting = False
    for frame_index in range(len(heights)):
        if contacting:
            contacting = bool(
                heights[frame_index] < exit_h and speeds[frame_index] < exit_v
            )
        else:
            contacting = bool(
                heights[frame_index] < enter_h and speeds[frame_index] < enter_v
            )
        result[frame_index] = contacting
    return result


def contact_masks(
    geometry: GeometrySeries,
    fps: float,
    part_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return hysteretic and median-smoothed contact masks (T, P)."""
    frame_count = geometry.heights.shape[0]
    hysteretic = np.zeros((frame_count, len(part_names)), dtype=bool)
    smoothed = np.zeros_like(hysteretic)
    median_width = _median_window_frames(fps)
    for part_index, part_name in enumerate(part_names):
        hysteretic[:, part_index] = _hysteresis_mask(
            geometry.heights[:, part_index],
            geometry.speeds[:, part_index],
            part_name,
        )
        smoothed[:, part_index] = _median_filter_bool(
            hysteretic[:, part_index], median_width
        )
    return hysteretic, smoothed


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _merge_and_filter_runs(
    runs: Iterable[tuple[int, int]],
    fps: float,
    min_window_s: float,
) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, stop in runs:
        if merged and (start - merged[-1][1]) / fps < MERGE_GAP_S:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return [
        (start, stop)
        for start, stop in merged
        if (stop - start) / fps >= min_window_s
    ]


def _summary(part_totals: dict[str, float]) -> str:
    if not part_totals:
        return "no contacts detected"
    ranked = sorted(part_totals.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{part} {seconds:.1f}s" for part, seconds in ranked[:6])


def _source_from_path(path: Path) -> str:
    lowered_parts = {part.lower() for part in path.parts}
    for source in ("aistpp", "finedance"):
        if source in lowered_parts:
            return source
    raise ValueError(f"{path}: path is not under an aistpp or finedance directory")


def _stem_from_path(path: Path) -> str:
    if not path.name.endswith(PKL_SUFFIX):
        raise ValueError(f"{path}: filename must end with {PKL_SUFFIX!r}")
    return path.name[: -len(PKL_SUFFIX)]


def build_clip_signature(
    path: str | Path,
    data: dict[str, Any],
    geometry: GeometrySeries,
    part_names: list[str],
) -> dict[str, Any]:
    """Convert measurements for one clip into the required JSON entry."""
    path = Path(path)
    fps = float(data["fps"])
    frame_count = geometry.heights.shape[0]
    _, smoothed = contact_masks(geometry, fps, part_names)
    windows: list[dict[str, Any]] = []
    part_totals: dict[str, float] = {}

    for part_index, part_name in enumerate(part_names):
        min_seconds = MIN_WINDOW_S.get(part_name, MIN_WINDOW_S["default"])
        runs = _merge_and_filter_runs(
            _true_runs(smoothed[:, part_index]), fps, min_seconds
        )
        if not runs:
            continue
        total_seconds = sum((stop - start) / fps for start, stop in runs)
        part_totals[part_name] = round(total_seconds, 3)
        for start, stop in runs:
            part_heights = geometry.heights[start:stop, part_index]
            part_speeds = geometry.speeds[start:stop, part_index]
            windows.append(
                {
                    "part": part_name,
                    "start_s": round(start / fps, 3),
                    "end_s": round(stop / fps, 3),
                    "min_height_m": round(float(part_heights.min()), 4),
                    "mean_height_m": round(float(part_heights.mean()), 4),
                    "mean_speed_ms": round(float(part_speeds.mean()), 3),
                }
            )

    windows.sort(key=lambda window: (window["start_s"], window["part"]))
    return {
        "source": _source_from_path(path),
        "fps": float(fps),
        "num_frames": int(frame_count),
        "duration_s": round(frame_count / fps, 2),
        "floor_offset_m": round(geometry.floor_offset_m, 3),
        "windows": windows,
        "part_totals": part_totals,
        "summary": _summary(part_totals),
    }


def _params_json(part_names: list[str]) -> dict[str, Any]:
    return {
        "h_enter_m": {part: H_ENTER[part] for part in part_names},
        "h_exit_extra_m": H_EXIT_EXTRA_M,
        "v_max_ms": {part: V_MAX[part] for part in part_names},
        "band_m": BAND_M,
        "median_win_s": MEDIAN_WIN_S,
        "merge_gap_s": MERGE_GAP_S,
        "min_window_s": dict(MIN_WINDOW_S),
        "parts": list(part_names),
    }


def _new_document(part_names: list[str]) -> dict[str, Any]:
    return {
        "_comment": (
            "Geometric mesh-vs-floor contact signatures produced by "
            "contact_geometric.py. Cleaned SMPL meshes are posed with numpy "
            "linear blend skinning; a body part contacts the measured floor "
            "when its lowest vertices are near the floor and its contact-band "
            "material vertices have low world-XZ velocity. Hysteresis, "
            "time-scaled median smoothing, gap merging, and minimum-duration "
            "filtering produce the final windows. This replaces the failed "
            "DECO learned-prior pilot."
        ),
        "method": "geometric_v1",
        "params": _params_json(part_names),
        "clips": {},
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def _partition_for(path: Path) -> tuple[np.ndarray, list[str]]:
    vertex_parts, part_names = build_vertex_parts(str(path))
    return np.asarray(vertex_parts, dtype=np.int64), list(part_names)


def _discover_library(root: Path) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    for source in ("aistpp", "finedance"):
        source_dir = root / source
        if not source_dir.is_dir():
            continue
        clips.extend(
            (source, path)
            for path in sorted(source_dir.glob(f"*{PKL_SUFFIX}"))
            if path.is_file()
        )
    return clips


def _find_clip(root: Path, stem: str) -> Path:
    if stem.endswith(PKL_SUFFIX):
        stem = stem[: -len(PKL_SUFFIX)]
    matches = [
        root / source / f"{stem}{PKL_SUFFIX}"
        for source in ("aistpp", "finedance")
        if (root / source / f"{stem}{PKL_SUFFIX}").is_file()
    ]
    if not matches:
        raise FileNotFoundError(
            f"no cleaned aistpp/finedance pickle found for stem {stem!r} under {root}"
        )
    if len(matches) > 1:
        raise ValueError(f"stem {stem!r} is ambiguous: {matches}")
    return matches[0]


def _mask_from_segments(
    segments: Iterable[tuple[int, int]], frame_count: int
) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for raw_start, raw_stop in segments:
        start = max(0, min(frame_count, int(raw_start)))
        stop = max(start, min(frame_count, int(raw_stop)))
        mask[start:stop] = True
    return mask


def _mask_from_windows(
    windows: Iterable[dict[str, Any]],
    parts: tuple[str, ...],
    frame_count: int,
    fps: float,
) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for window in windows:
        if window.get("part") not in parts:
            continue
        start = max(0, min(frame_count, int(round(float(window["start_s"]) * fps))))
        stop = max(
            start,
            min(frame_count, int(round(float(window["end_s"]) * fps))),
        )
        mask[start:stop] = True
    return mask


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _format_ratio(value: float | None) -> str:
    return "  n/a" if value is None else f"{100.0 * value:5.1f}"


def validation_report(
    document: dict[str, Any],
    paths_by_stem: dict[str, Path],
    part_names: list[str],
) -> str:
    """Compare geometric windows with cleanup joint-contact segments."""
    lines = [
        "Geometric contact validation",
        "============================",
        (
            "Each cell is recall/precision (%): cleanup-contact time covered by "
            "geometric windows / geometric-window time covered by cleanup."
        ),
        "Wrist predictions use palm + forearm on the matching side.",
        "",
        (
            f"{'clip':36s} {'L ankle':>13s} {'R ankle':>13s} "
            f"{'L wrist':>13s} {'R wrist':>13s} {'ALL':>13s}"
        ),
        "-" * 110,
    ]

    time_totals = {
        effector: {"cleanup": 0.0, "geometric": 0.0, "overlap": 0.0}
        for effector in EFFECTOR_PARTS
    }

    for stem, clip in sorted(document["clips"].items()):
        path = paths_by_stem[stem]
        data = _load_pkl(path)
        fps = float(data["fps"])
        frame_count = int(data["num_frames"])
        cleanup_segments = data.get("cleanup", {}).get("segments", {})
        cells: list[str] = []
        clip_cleanup = 0
        clip_geometric = 0
        clip_overlap = 0

        for effector, prediction_parts in EFFECTOR_PARTS.items():
            cleanup_mask = _mask_from_segments(
                cleanup_segments.get(effector, []), frame_count
            )
            geometric_mask = _mask_from_windows(
                clip["windows"], prediction_parts, frame_count, fps
            )
            overlap = int(np.count_nonzero(cleanup_mask & geometric_mask))
            cleanup_count = int(np.count_nonzero(cleanup_mask))
            geometric_count = int(np.count_nonzero(geometric_mask))
            recall = _ratio(overlap, cleanup_count)
            precision = _ratio(overlap, geometric_count)
            cells.append(f"{_format_ratio(recall)}/{_format_ratio(precision)}")

            time_totals[effector]["cleanup"] += cleanup_count / fps
            time_totals[effector]["geometric"] += geometric_count / fps
            time_totals[effector]["overlap"] += overlap / fps
            clip_cleanup += cleanup_count
            clip_geometric += geometric_count
            clip_overlap += overlap

        overall = (
            f"{_format_ratio(_ratio(clip_overlap, clip_cleanup))}/"
            f"{_format_ratio(_ratio(clip_overlap, clip_geometric))}"
        )
        lines.append(
            f"{stem[:36]:36s} " + " ".join(f"{cell:>13s}" for cell in cells)
            + f" {overall:>13s}"
        )

    lines.extend(
        [
            "",
            "Library totals by cleanup effector",
            "----------------------------------",
            (
                f"{'effector':12s} {'cleanup_s':>10s} {'geometric_s':>12s} "
                f"{'overlap_s':>10s} {'recall_%':>10s} {'precision_%':>12s}"
            ),
        ]
    )
    all_time = {"cleanup": 0.0, "geometric": 0.0, "overlap": 0.0}
    for effector, values in time_totals.items():
        recall = values["overlap"] / values["cleanup"] if values["cleanup"] else None
        precision = (
            values["overlap"] / values["geometric"]
            if values["geometric"]
            else None
        )
        lines.append(
            f"{effector:12s} {values['cleanup']:10.2f} {values['geometric']:12.2f} "
            f"{values['overlap']:10.2f} {_format_ratio(recall):>10s} "
            f"{_format_ratio(precision):>12s}"
        )
        for key in all_time:
            all_time[key] += values[key]
    all_recall = (
        all_time["overlap"] / all_time["cleanup"] if all_time["cleanup"] else None
    )
    all_precision = (
        all_time["overlap"] / all_time["geometric"]
        if all_time["geometric"]
        else None
    )
    lines.append(
        f"{'ALL':12s} {all_time['cleanup']:10.2f} {all_time['geometric']:12.2f} "
        f"{all_time['overlap']:10.2f} {_format_ratio(all_recall):>10s} "
        f"{_format_ratio(all_precision):>12s}"
    )

    lines.extend(
        [
            "",
            "Library-wide geometric contacts by part",
            "---------------------------------------",
            f"{'part':14s} {'clips_fired':>12s} {'total_s':>12s}",
        ]
    )
    for part_name in part_names:
        fired = sum(
            1
            for clip in document["clips"].values()
            if float(clip["part_totals"].get(part_name, 0.0)) > 0.0
        )
        total_seconds = sum(
            float(clip["part_totals"].get(part_name, 0.0))
            for clip in document["clips"].values()
        )
        lines.append(f"{part_name:14s} {fired:12d} {total_seconds:12.3f}")
    return "\n".join(lines) + "\n"


def run_stats(path: Path) -> None:
    data = _load_pkl(path)
    vertex_parts, part_names = _partition_for(path)
    geometry = compute_geometry(data, vertex_parts, part_names, label=str(path))
    fps = float(data["fps"])
    print(f"pkl: {path}")
    print(
        f"frames={data['num_frames']} fps={fps:g} "
        f"duration={int(data['num_frames']) / fps:.2f}s"
    )
    print(
        f"measured_floor_y={geometry.floor_offset_m:+.4f} m "
        f"floor_y_used={geometry.floor_y_used:+.4f} m"
    )
    print(
        f"{'part':14s} {'h01':>8s} {'h05':>8s} {'h25':>8s} {'h50':>8s} "
        f"{'v01':>8s} {'v05':>8s} {'v25':>8s} {'v50':>8s} {'low_n':>8s}"
    )
    for part_index, part_name in enumerate(part_names):
        heights = geometry.heights[:, part_index]
        h_pct = np.percentile(heights, (1, 5, 25, 50))
        low = heights < 0.10
        if np.any(low):
            v_pct = np.percentile(geometry.speeds[low, part_index], (1, 5, 25, 50))
        else:
            v_pct = np.full(4, np.nan)
        values = " ".join(f"{value:8.4f}" for value in (*h_pct, *v_pct))
        print(f"{part_name:14s} {values} {int(low.sum()):8d}")


def run_build(path: Path, merge_into: Path | None) -> None:
    data = _load_pkl(path)
    vertex_parts, part_names = _partition_for(path)
    geometry = compute_geometry(data, vertex_parts, part_names, label=str(path))
    signature = build_clip_signature(path, data, geometry, part_names)
    stem = _stem_from_path(path)
    print(json.dumps(signature["windows"], indent=2, ensure_ascii=False))
    print(f"[{stem}] {signature['summary']}")

    if merge_into is not None:
        document = _new_document(part_names)
        if merge_into.exists():
            with merge_into.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("method") == "geometric_v1":
                document["clips"].update(existing.get("clips", {}))
            else:
                print(
                    f"WARNING: replacing non-geometric contact file {merge_into}",
                    file=sys.stderr,
                )
        document["clips"][stem] = signature
        _write_json(merge_into, document)
        print(f"merged into {merge_into}")


def run_library(root: Path, out: Path, stride: int) -> None:
    discovered = _discover_library(root)
    if not discovered:
        raise FileNotFoundError(
            f"no {PKL_SUFFIX} files found under {root / 'aistpp'} or {root / 'finedance'}"
        )
    print(f"Found {len(discovered)} cleaned clips under {root}")
    canonical_path = discovered[0][1]
    vertex_parts, part_names = _partition_for(canonical_path)
    document = _new_document(part_names)
    paths_by_stem: dict[str, Path] = {}

    for index, (source, path) in enumerate(discovered, start=1):
        stem = _stem_from_path(path)
        if stem in paths_by_stem:
            raise ValueError(f"duplicate library stem {stem!r}")
        print(f"[{index:02d}/{len(discovered):02d}] {source}/{path.name}")
        data = _load_pkl(path)
        geometry = compute_geometry(
            data,
            vertex_parts,
            part_names,
            stride=stride,
            label=f"{source}/{stem}",
        )
        signature = build_clip_signature(path, data, geometry, part_names)
        document["clips"][stem] = signature
        paths_by_stem[stem] = path
        print(f"  {signature['summary']}")

    _write_json(out, document)
    print(f"Wrote {out}")
    report = validation_report(document, paths_by_stem, part_names)
    validation_path = root / DEFAULT_VALIDATION_OUT.name
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(report, encoding="utf-8", newline="\n")
    print()
    print(report, end="")
    print(f"Wrote {validation_path}")


def run_inspect(
    root: Path,
    stem: str,
    part_name: str,
    from_s: float,
    to_s: float | None,
) -> None:
    path = _find_clip(root, stem)
    data = _load_pkl(path)
    vertex_parts, part_names = _partition_for(path)
    geometry = compute_geometry(data, vertex_parts, part_names, label=str(path))
    fps = float(data["fps"])
    duration = int(data["num_frames"]) / fps
    if from_s < 0.0:
        raise ValueError("--from must be non-negative")
    if to_s is None:
        to_s = duration
    if to_s < from_s:
        raise ValueError("--to must be greater than or equal to --from")
    start = max(0, min(int(data["num_frames"]), int(np.floor(from_s * fps))))
    stop = max(start, min(int(data["num_frames"]), int(np.ceil(to_s * fps))))
    part_index = part_names.index(part_name)
    hysteretic, smoothed = contact_masks(geometry, fps, part_names)
    raw = (
        (geometry.heights[:, part_index] < H_ENTER[part_name])
        & (geometry.speeds[:, part_index] < V_MAX[part_name])
    )

    print(f"pkl: {path}")
    print(
        f"part={part_name} range={start / fps:.3f}..{stop / fps:.3f}s "
        f"floor_offset={geometry.floor_offset_m:+.4f}m"
    )
    print(f"{'frame':>7s} {'time_s':>9s} {'h_m':>9s} {'v_ms':>9s} raw hyst smooth")
    for frame_index in range(start, stop):
        print(
            f"{frame_index:7d} {frame_index / fps:9.3f} "
            f"{geometry.heights[frame_index, part_index]:9.4f} "
            f"{geometry.speeds[frame_index, part_index]:9.4f} "
            f" {int(raw[frame_index])}    {int(hysteretic[frame_index, part_index])}"
            f"     {int(smoothed[frame_index, part_index])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-numpy geometric SMPL mesh-vs-floor contact detection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stats_parser = subparsers.add_parser(
        "stats", help="print per-part height and low-frame velocity distributions"
    )
    stats_parser.add_argument("pkl", type=Path)

    build_parser_ = subparsers.add_parser(
        "build", help="build and print one clip signature"
    )
    build_parser_.add_argument("pkl", type=Path)
    build_parser_.add_argument(
        "--merge-into", type=Path, default=None, metavar="JSON"
    )

    library_parser = subparsers.add_parser(
        "library", help="build all AIST++ and FineDance signatures and validate"
    )
    library_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    library_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    library_parser.add_argument(
        "--stride",
        type=_positive_int,
        default=1,
        help="use every Nth vertex for part detection (floor still uses all vertices)",
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="print per-frame height and speed for a library stem"
    )
    inspect_parser.add_argument("stem")
    inspect_parser.add_argument(
        "--part", choices=list(PART_GROUPS.keys()), default="foot_L"
    )
    inspect_parser.add_argument("--from", dest="from_s", type=float, default=0.0)
    inspect_parser.add_argument("--to", dest="to_s", type=float, default=None)
    inspect_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "stats":
            run_stats(args.pkl)
        elif args.command == "build":
            run_build(args.pkl, args.merge_into)
        elif args.command == "library":
            run_library(args.root, args.out, args.stride)
        elif args.command == "inspect":
            run_inspect(args.root, args.stem, args.part, args.from_s, args.to_s)
        else:  # pragma: no cover - argparse enforces the choices
            raise AssertionError(f"unexpected command {args.command!r}")
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
