"""Compile affordance scripts into independent DART scene interactions.

Affordance compiler v0 turns a Y-up scene plus an ordered list of stations
into one cropped mesh2sdf volume and one DART interaction JSON per segment.
The accepted script JSON is:

    {
      "title": str,
      "scene": str,  # repo-relative or absolute GLB path; Y-up, meters
      "start": {
        "at": [x, z],
        "heading_deg": float  # optional
      },
      "defaults": {  # optional, as are all of its fields
        "voxel_m": 0.02,
        "corridor_halfwidth_m": 1.0,
        "margin_m": 0.75,
        "max_grid": 256
      },
      "stations": [
        {
          "label": str,
          "at": [x, z],  # Y-up XZ
          "heading_deg": float,  # optional absolute yaw
          "face_travel": bool,  # optional, default true
          "text": str,  # prompt for the segment ending here
          "surface_z": float,  # optional support height, default 0 m
          "target_object": str,  # optional metadata hint
          "contact_ref": str  # optional pass-through metadata key
        }
      ]
    }

N stations produce N independent DART runs. Segment zero starts at ``start``;
later segments start at the preceding station. Normally the initial body faces
the travel direction. If the segment's start station explicitly sets
``face_travel`` false and supplies ``heading_deg``, that absolute heading is
used instead. Travel shorter than 0.15 m uses the start station heading, or
zero degrees when absent. Heading follows :mod:`choreo`: ``atan2(dx, dz)`` in
Y-up XZ, so 0 degrees faces +Z and 90 degrees faces +X.

glTF scenes use Y-up while DART/AMASS use Z-up. The exact conversion is
``(x, y, z) -> (x, -z, y)``; heights therefore map one-to-one. Each GLB
geometry keeps its scene-graph transform, is converted to Z-up, and is sliced
against the segment crop. The SDF uses ``mesh_scale=0.8``, ``fix=True``, and
``level=2/size``. This deliberately differs from the rehearsal's DART
``level=1/size`` recipe: mesh2sdf's fix path keeps only the largest connected
level-set component, and the larger documented level prevents thin slabs from
fragmenting and flooding positive. The fixed SDF measures distance to a level
set about ``level`` outside the true surface, so obstacles appear about one
voxel fatter; the rehearsal's global grid had the same-magnitude bias at
3.9 cm. Its center and scale come from the requested crop bbox, not the cropped
mesh's tight bounds. The normalized SDF grid is cubic: it covers the crop in
its largest dimension and extends beyond the bbox in smaller dimensions (cube
side is ``max_bbox_extent / 0.8``). Metric SDF values are normalized values /
scale.

The embedded standing pose comes from frame 0 of
``output/dart_rehearsal/experiment/b_sdf/sample_0.pkl``: the rehearsal's
``stand.pkl`` -> SMPL-X male zero-beta init constraint, yaw-unrotated by
-38.65977559 degrees to face +X. ``derive_canonical_pose`` recomputes the
pelvis height and hip offsets from that sample and handles pickled PosixPath
objects on Windows. Compilation checks the sample against the constants when
it is available, and otherwise warns and uses the constants.

Usage::

    python analyzer_3d/src/affordance_compile.py compile --script spec.json
    python analyzer_3d/src/affordance_compile.py validate --plan plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = Path("output/affordance_compile")
DEFAULT_VOLUME_ROOT = "/vol/scenes"
DEFAULTS = {
    "voxel_m": 0.02,
    "corridor_halfwidth_m": 1.0,
    "margin_m": 0.75,
    "max_grid": 256,
}

YUP_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, -1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=float,
)

CANON_PELVIS_Z = 0.9772447347640991
CANON_LHIP_OFF = (
    -0.013783369291996828,
    0.05678698056548785,
    -0.09605669975280762,
)
CANON_RHIP_OFF = (
    -0.013783369291996837,
    -0.05751800704629549,
    -0.1057557463645935,
)

INTERACTION_KEYS = {"scene_dir", "floor_height", "interaction_name", "interactions"}
INTERACTION_ITEM_KEYS = {"text_prompt", "init_joints", "goal_joints"}
SDF_CFG_KEYS = {"center", "scale", "level", "size"}


def _as_numpy(value: Any) -> np.ndarray:
    """Convert numpy arrays or CPU torch tensors without importing torch here."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def derive_canonical_pose(pkl_path: str | Path) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    """Derive the +X-facing frame-0 pelvis height and hip offsets.

    The pickle must contain ``joints`` shaped ``[T, >=3, 3]`` in Z-up world
    coordinates with joints 0:3 ordered pelvis, left hip, right hip. Torch is
    imported lazily so tensor-backed rehearsal samples can be unpickled.
    """
    try:
        import torch  # noqa: F401  # registers tensor pickle reconstruction
    except ImportError as exc:
        raise RuntimeError("torch is required to unpickle the rehearsal sample") from exc

    import pathlib as pathlib_module

    original_posix_path = pathlib_module.PosixPath
    if os.name == "nt":
        pathlib_module.PosixPath = pathlib_module.WindowsPath
    try:
        with Path(pkl_path).open("rb") as handle:
            sample = pickle.load(handle)
    finally:
        pathlib_module.PosixPath = original_posix_path

    joints = _as_numpy(sample["joints"])
    if joints.ndim != 3 or joints.shape[0] < 1 or joints.shape[1] < 3 or joints.shape[2] != 3:
        raise ValueError(f"Expected joints [T,>=3,3], got {joints.shape}")

    pelvis, lhip, rhip = (joints[0, i].copy() for i in range(3))
    hip_line = rhip - lhip
    facing = np.array([-hip_line[1], hip_line[0], 0.0], dtype=float)
    norm = float(np.linalg.norm(facing[:2]))
    if norm < 1e-10:
        raise ValueError("Cannot derive facing from coincident frame-0 hips")
    facing /= norm
    yaw = math.atan2(float(facing[1]), float(facing[0]))
    c, s = math.cos(-yaw), math.sin(-yaw)
    unrotate = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    lhip_off = unrotate @ (lhip - pelvis)
    rhip_off = unrotate @ (rhip - pelvis)
    return float(pelvis[2]), tuple(lhip_off.tolist()), tuple(rhip_off.tolist())


def _verify_canonical_pose() -> str:
    sample_path = REPO_ROOT / "output/dart_rehearsal/experiment/b_sdf/sample_0.pkl"
    if not sample_path.exists():
        message = f"Canonical rehearsal sample missing at {sample_path}; using embedded constants"
        warnings.warn(message, RuntimeWarning)
        return message
    try:
        derived = derive_canonical_pose(sample_path)
    except Exception as exc:  # unreadable pickle or unavailable tensor support
        message = f"Could not read canonical rehearsal sample ({exc}); using embedded constants"
        warnings.warn(message, RuntimeWarning)
        return message

    expected = np.array([CANON_PELVIS_Z, *CANON_LHIP_OFF, *CANON_RHIP_OFF])
    actual = np.array([derived[0], *derived[1], *derived[2]])
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-4,
        err_msg="Rehearsal-derived canonical pose disagrees with embedded constants",
    )
    return f"Verified embedded canonical pose against {sample_path.as_posix()} (atol=1e-4)"


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{where} must be a finite number")
    return result


def _xz(value: Any, where: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{where} must be a two-number JSON array [x, z]")
    return [_finite_number(value[0], f"{where}[0]"), _finite_number(value[1], f"{where}[1]")]


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown)}")


def _validate_script(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Affordance script root must be a JSON object")
    _reject_unknown(raw, {"title", "scene", "start", "defaults", "stations"}, "script")
    for key in ("title", "scene", "start", "stations"):
        if key not in raw:
            raise ValueError(f"Missing required script key: {key}")
    if not isinstance(raw["title"], str) or not raw["title"].strip():
        raise ValueError("title must be a non-empty string")
    if not isinstance(raw["scene"], str) or not raw["scene"].strip():
        raise ValueError("scene must be a non-empty path string")

    start = raw["start"]
    if not isinstance(start, dict):
        raise ValueError("start must be an object")
    _reject_unknown(start, {"at", "heading_deg"}, "start")
    if "at" not in start:
        raise ValueError("start.at is required")
    _xz(start["at"], "start.at")
    if "heading_deg" in start:
        _finite_number(start["heading_deg"], "start.heading_deg")

    supplied_defaults = raw.get("defaults", {})
    if not isinstance(supplied_defaults, dict):
        raise ValueError("defaults must be an object")
    _reject_unknown(supplied_defaults, set(DEFAULTS), "defaults")
    if ("voxel_m" in supplied_defaults
            and _finite_number(supplied_defaults["voxel_m"], "defaults.voxel_m") <= 0):
        raise ValueError("defaults.voxel_m must be > 0")
    for key in ("corridor_halfwidth_m", "margin_m"):
        if key in supplied_defaults and _finite_number(supplied_defaults[key], f"defaults.{key}") < 0:
            raise ValueError(f"defaults.{key} must be >= 0")
    if "max_grid" in supplied_defaults:
        value = supplied_defaults["max_grid"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 64:
            raise ValueError("defaults.max_grid must be an integer >= 64")

    stations = raw["stations"]
    if not isinstance(stations, list) or not stations:
        raise ValueError("stations must be a non-empty array")
    station_keys = {
        "label", "at", "heading_deg", "face_travel", "text", "surface_z",
        "target_object", "contact_ref",
    }
    for index, station in enumerate(stations):
        where = f"stations[{index}]"
        if not isinstance(station, dict):
            raise ValueError(f"{where} must be an object")
        _reject_unknown(station, station_keys, where)
        for key in ("label", "at", "text"):
            if key not in station:
                raise ValueError(f"{where}.{key} is required")
        if not isinstance(station["label"], str) or not station["label"].strip():
            raise ValueError(f"{where}.label must be a non-empty string")
        if not isinstance(station["text"], str) or not station["text"].strip():
            raise ValueError(f"{where}.text must be a non-empty string")
        _xz(station["at"], f"{where}.at")
        if "heading_deg" in station:
            _finite_number(station["heading_deg"], f"{where}.heading_deg")
        if "surface_z" in station:
            _finite_number(station["surface_z"], f"{where}.surface_z")
        if "face_travel" in station and not isinstance(station["face_travel"], bool):
            raise ValueError(f"{where}.face_travel must be boolean")
        for key in ("target_object", "contact_ref"):
            if key in station and (not isinstance(station[key], str) or not station[key].strip()):
                raise ValueError(f"{where}.{key} must be a non-empty string")
    return raw


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _path_text(path: Path) -> str:
    return path.resolve().as_posix()


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "station"


def _zup_xy(yup_xz: Iterable[float]) -> np.ndarray:
    x, z = yup_xz
    return np.array([float(x), -float(z)], dtype=float)


def _heading_facing_zup(heading_deg: float) -> np.ndarray:
    theta = math.radians(float(heading_deg))
    # Y-up (sin(theta), cos(theta)) in XZ -> Z-up (sin(theta), -cos(theta)) in XY.
    return np.array([math.sin(theta), -math.cos(theta)], dtype=float)


def _segment_facing(start_station: dict[str, Any], start_xy: np.ndarray,
                    goal_xy: np.ndarray) -> tuple[np.ndarray, str]:
    delta = goal_xy - start_xy
    distance = float(np.linalg.norm(delta))
    heading = float(start_station.get("heading_deg", 0.0))
    if start_station.get("face_travel", True) is False and "heading_deg" in start_station:
        return _heading_facing_zup(heading), "start station absolute heading"
    if distance < 0.15:
        return _heading_facing_zup(heading), "short-segment heading fallback"
    return delta / distance, "travel direction"


def _standing_joints(start_xy: np.ndarray, facing: np.ndarray,
                     start_surface_z: float, goal_xy: np.ndarray,
                     goal_surface_z: float) -> tuple[list[list[float]], list[list[float]]]:
    angle = math.atan2(float(facing[1]), float(facing[0]))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pelvis = np.array([start_xy[0], start_xy[1], CANON_PELVIS_Z + start_surface_z])
    init = [
        pelvis,
        rotation @ np.asarray(CANON_LHIP_OFF) + pelvis,
        rotation @ np.asarray(CANON_RHIP_OFF) + pelvis,
    ]
    goal = [[float(goal_xy[0]), float(goal_xy[1]), CANON_PELVIS_Z + goal_surface_z]]
    return [joint.tolist() for joint in init], goal


def _text_prompt(text: str, distance_m: float) -> str:
    if "*" in text:
        return text
    repeats = min(60, max(6, math.ceil(distance_m * 3.75)))
    return f"{text}*{repeats}"


def _load_scene_objects(scene_path: Path) -> list[dict[str, Any]]:
    """Load a GLB without forcing a mesh and bake every instance transform."""
    import trimesh

    loaded = trimesh.load(str(scene_path))
    objects: list[dict[str, Any]] = []
    if isinstance(loaded, trimesh.Scene):
        for node_name in sorted(loaded.graph.nodes_geometry, key=str):
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name]
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            mesh = geometry.copy()
            mesh.apply_transform(np.asarray(transform, dtype=float))
            mesh.apply_transform(YUP_TO_ZUP)
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            objects.append({
                "name": f"{geometry_name}@{node_name}",
                "mesh": mesh,
                "bounds": np.asarray(mesh.bounds, dtype=float),
            })
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded.copy()
        mesh.apply_transform(YUP_TO_ZUP)
        if len(mesh.vertices) and len(mesh.faces):
            objects.append({"name": scene_path.stem, "mesh": mesh,
                            "bounds": np.asarray(mesh.bounds, dtype=float)})
    else:
        raise TypeError(f"Expected trimesh.Scene or Trimesh from {scene_path}, got {type(loaded)}")
    if not objects:
        raise ValueError(f"Scene contains no triangle geometries: {scene_path}")
    return objects


def _crop_bbox(objects: list[dict[str, Any]], start_xy: np.ndarray,
               goal_xy: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    xy_min = np.minimum(start_xy, goal_xy) - padding
    xy_max = np.maximum(start_xy, goal_xy) + padding
    scene_zmin = min(float(obj["bounds"][0, 2]) for obj in objects)
    overlapping_tops = [
        float(obj["bounds"][1, 2]) for obj in objects
        if np.all(obj["bounds"][1, :2] >= xy_min)
        and np.all(obj["bounds"][0, :2] <= xy_max)
    ]
    tallest = max(overlapping_tops) if overlapping_tops else scene_zmin
    bbox_min = np.array([xy_min[0], xy_min[1], scene_zmin - 0.05], dtype=float)
    bbox_max = np.array([xy_max[0], xy_max[1], max(2.5, tallest + 0.3)], dtype=float)
    return bbox_min, bbox_max


def _crop_scene_mesh(objects: list[dict[str, Any]], bbox_min: np.ndarray,
                     bbox_max: np.ndarray) -> tuple[Any, list[str]]:
    """Slice overlapping objects to six inward-facing bbox planes."""
    import trimesh

    kept = []
    crop_warnings: list[str] = []
    planes = [
        (np.array([bbox_min[0], 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([bbox_max[0], 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
        (np.array([0.0, bbox_min[1], 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, bbox_max[1], 0.0]), np.array([0.0, -1.0, 0.0])),
        (np.array([0.0, 0.0, bbox_min[2]]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, bbox_max[2]]), np.array([0.0, 0.0, -1.0])),
    ]
    for obj in objects:
        bounds = obj["bounds"]
        if np.any(bounds[1] < bbox_min) or np.any(bounds[0] > bbox_max):
            continue
        if np.all(bounds[0] >= bbox_min) and np.all(bounds[1] <= bbox_max):
            kept.append(obj["mesh"].copy())
            continue
        try:
            sliced = obj["mesh"].copy()
            for origin, normal in planes:
                sliced = sliced.slice_plane(origin, normal, cap=True)
                if sliced is None or len(sliced.vertices) == 0 or len(sliced.faces) == 0:
                    raise ValueError("slice returned an empty mesh")
            kept.append(sliced)
        except Exception as exc:
            kept.append(obj["mesh"].copy())
            crop_warnings.append(
                f"{obj['name']}: bbox slicing failed ({type(exc).__name__}: {exc}); "
                "included the whole overlapping geometry"
            )
    if not kept:
        raise ValueError("Crop contains no scene geometry")
    return trimesh.util.concatenate(kept), crop_warnings


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _grid_size(max_extent: float, voxel_m: float, max_grid: int) -> tuple[int, int]:
    needed = int(math.ceil(max_extent / (0.8 * voxel_m)))
    target = _next_power_of_two(needed)
    return max(64, min(target, max_grid)), needed


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_sdf(cropped_mesh: Any, bbox_min: np.ndarray, bbox_max: np.ndarray,
              size: int, scene_dir: Path, crop_npz: Path) -> tuple[dict[str, Any], float]:
    import mesh2sdf

    extent = bbox_max - bbox_min
    max_extent = float(extent.max())
    center = (bbox_min + bbox_max) * 0.5
    scale = 2.0 * 0.8 / max_extent
    level = 2.0 / size
    sdf, _fixed = mesh2sdf.compute(
        (np.asarray(cropped_mesh.vertices) - center) * scale,
        np.asarray(cropped_mesh.faces),
        size,
        fix=True,
        level=level,
        return_mesh=True,
    )
    sdf = np.asarray(sdf, dtype=np.float32)
    cfg = {"center": center.tolist(), "scale": float(scale),
           "level": float(level), "size": int(size)}
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "scene_sdf.npy", sdf)
    _write_json(scene_dir / "scene_sdf.json", cfg)
    cropped_mesh.export(scene_dir / "scene_with_floor.obj")
    voxel_m_actual = max_extent / (0.8 * size)
    np.savez_compressed(
        crop_npz,
        sdf=sdf,
        center=np.asarray(center, dtype=float),
        scale=np.float64(scale),
        level=np.float64(level),
        size=np.int32(size),
        bbox_min_zup=np.asarray(bbox_min, dtype=float),
        bbox_max_zup=np.asarray(bbox_max, dtype=float),
        voxel_m_actual=np.float64(voxel_m_actual),
    )
    return cfg, float(voxel_m_actual)


def calc_point_sdf(grid: np.ndarray, cfg: dict[str, Any],
                   points_zup: np.ndarray) -> np.ndarray:
    """Return metric SDF values, exactly mirroring dart_modal._calc_sdf_cpu."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for SDF validation") from exc

    sdf_grid = torch.as_tensor(grid, dtype=torch.float32)[None]
    points = torch.as_tensor(points_zup, dtype=torch.float32)[None]
    scale = float(cfg["scale"])
    center = torch.tensor(cfg["center"], dtype=torch.float32).view(1, 1, 3)
    normalized = (points - center) * scale
    values = functional.grid_sample(
        sdf_grid.unsqueeze(0),
        normalized[:, :, [2, 1, 0]].view(1, -1, 1, 1, 3),
        padding_mode="border",
        align_corners=True,
    ).reshape(-1)
    return (values / scale).numpy()


def _schema_row(segment: str, name: str, verdict: str, details: str) -> dict[str, Any]:
    return {"segment": segment, "probe": name, "point_zup": None,
            "expected": "exact schema", "value_m": None,
            "verdict": verdict, "details": details}


def _check_schemas(plan_dir: Path, segment: dict[str, Any]) -> tuple[list[dict[str, Any]],
                                                                        np.ndarray | None,
                                                                        dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    seg_id = segment["seg_id"]
    artifacts = segment["artifacts"]
    try:
        interaction = json.loads((plan_dir / artifacts["interaction_json"]).read_text(encoding="utf-8"))
        assert set(interaction) == INTERACTION_KEYS, f"keys={sorted(interaction)}"
        assert isinstance(interaction["interactions"], list) and len(interaction["interactions"]) == 1
        item = interaction["interactions"][0]
        assert set(item) == INTERACTION_ITEM_KEYS, f"interaction keys={sorted(item)}"
        assert np.asarray(item["init_joints"], dtype=float).shape == (3, 3)
        assert np.asarray(item["goal_joints"], dtype=float).shape == (1, 3)
        rows.append(_schema_row(seg_id, "interaction schema", "PASS", "exact keys and joint shapes"))
    except Exception as exc:
        rows.append(_schema_row(seg_id, "interaction schema", "FAIL", str(exc)))

    grid = None
    cfg = None
    try:
        cfg = json.loads((plan_dir / artifacts["scene_sdf_json"]).read_text(encoding="utf-8"))
        grid = np.load(plan_dir / artifacts["scene_sdf_npy"])
        assert set(cfg) == SDF_CFG_KEYS, f"keys={sorted(cfg)}"
        size = int(cfg["size"])
        assert float(cfg["level"]) == 2.0 / size, "level must equal 2.0/size"
        assert grid.shape == (size, size, size), f"grid shape={grid.shape}, size={size}"
        rows.append(_schema_row(seg_id, "SDF schema", "PASS", "exact config keys, level, and grid shape"))
    except Exception as exc:
        grid = None
        cfg = None
        rows.append(_schema_row(seg_id, "SDF schema", "FAIL", str(exc)))
    return rows, grid, cfg


def _inside_aabb(point: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(np.all(point >= bounds[0]) and np.all(point <= bounds[1]))


def _sign_verdict(value_m: float, expected: str, tolerance: float) -> str:
    correct = value_m < 0.0 if expected == "NEGATIVE" else value_m > 0.0
    if abs(value_m) <= tolerance:
        return "WARN"
    return "PASS" if correct else "FAIL"


def _probe_row(segment: str, name: str, point: np.ndarray, expected: str,
               value_m: float | None, verdict: str, details: str = "") -> dict[str, Any]:
    return {
        "segment": segment,
        "probe": name,
        "point_zup": [float(value) for value in point],
        "expected": expected,
        "value_m": None if value_m is None else float(value_m),
        "verdict": verdict,
        "details": details,
    }


def _run_probes(segment: dict[str, Any], objects: list[dict[str, Any]],
                grid: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seg_id = segment["seg_id"]
    bbox_min = np.asarray(segment["crop_bbox_zup"]["min"], dtype=float)
    bbox_max = np.asarray(segment["crop_bbox_zup"]["max"], dtype=float)
    interior_min = bbox_min + 0.1
    interior_max = bbox_max - 0.1
    tolerance = 0.5 * float(segment["voxel_m_actual"])
    pending: list[tuple[str, np.ndarray, str]] = []
    rows: list[dict[str, Any]] = []

    for obj in objects:
        center = obj["bounds"].mean(axis=0)
        if not (np.all(center > interior_min) and np.all(center < interior_max)):
            continue
        pending.append((f"{obj['name']} center", center, "NEGATIVE"))
        air = np.array([center[0], center[1], obj["bounds"][1, 2] + 0.3])
        if any(_inside_aabb(air, other["bounds"]) for other in objects):
            rows.append(_probe_row(seg_id, f"{obj['name']} above top", air, "POSITIVE",
                                   None, "SKIP", "point lies inside another object AABB"))
        else:
            pending.append((f"{obj['name']} above top", air, "POSITIVE"))

    start = np.asarray(segment["start_zup_xy"], dtype=float)
    goal = np.asarray(segment["goal_zup_xy"], dtype=float)
    midpoint_xy = (start + goal) * 0.5
    surfaces = segment["surface_z"]
    route_air = np.array([midpoint_xy[0], midpoint_xy[1],
                          0.9 + max(float(surfaces["start"]), float(surfaces["goal"]))])
    if any(_inside_aabb(route_air, obj["bounds"]) for obj in objects):
        rows.append(_probe_row(seg_id, "route midpoint body height", route_air, "POSITIVE",
                               None, "SKIP", "point lies inside an object AABB"))
    else:
        pending.append(("route midpoint body height", route_air, "POSITIVE"))

    below_floor = np.array([midpoint_xy[0], midpoint_xy[1], -0.1])
    if any(_inside_aabb(below_floor, obj["bounds"]) for obj in objects):
        pending.append(("route midpoint below floor", below_floor, "NEGATIVE"))
    else:
        rows.append(_probe_row(seg_id, "route midpoint below floor", below_floor, "NEGATIVE",
                               None, "SKIP", "no original-scene object AABB occupies the point"))

    if pending:
        values = calc_point_sdf(grid, cfg, np.asarray([item[1] for item in pending]))
        for (name, point, expected), value in zip(pending, values):
            metric = float(value)
            rows.append(_probe_row(seg_id, name, point, expected, metric,
                                   _sign_verdict(metric, expected, tolerance)))
    return rows


def _render_slice(plan_dir: Path, segment: dict[str, Any], grid: np.ndarray,
                  cfg: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    size = int(cfg["size"])
    center = np.asarray(cfg["center"], dtype=float)
    scale = float(cfg["scale"])
    surfaces = segment["surface_z"]
    desired_z = 0.6 + max(float(surfaces["start"]), float(surfaces["goal"]))
    normalized_z = float(np.clip((desired_z - center[2]) * scale, -1.0, 1.0))
    z_index = int(round((normalized_z + 1.0) * 0.5 * (size - 1)))
    actual_z = center[2] + ((2.0 * z_index / (size - 1)) - 1.0) / scale
    sdf_slice = np.asarray(grid[:, :, z_index], dtype=float).T / scale

    x_min, x_max = center[0] - 1.0 / scale, center[0] + 1.0 / scale
    y_min, y_max = center[1] - 1.0 / scale, center[1] + 1.0 / scale
    limit = max(float(np.max(np.abs(sdf_slice))), float(segment["voxel_m_actual"]), 1e-6)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(
        sdf_slice,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
    )
    xs = np.linspace(x_min, x_max, size)
    ys = np.linspace(y_min, y_max, size)
    if float(sdf_slice.min()) <= 0.0 <= float(sdf_slice.max()):
        ax.contour(xs, ys, sdf_slice, levels=[0.0], colors="black", linewidths=0.8)
    fig.colorbar(image, ax=ax, label="SDF [m]")

    start = np.asarray(segment["start_zup_xy"], dtype=float)
    goal = np.asarray(segment["goal_zup_xy"], dtype=float)
    ax.plot([start[0], goal[0]], [start[1], goal[1]], "k--", linewidth=1.2, label="route")
    ax.plot(start[0], start[1], marker="o", color="limegreen", mec="black", label="start")
    ax.plot(goal[0], goal[1], marker="*", color="gold", mec="black", ms=12, label="goal")
    bbox_min = np.asarray(segment["crop_bbox_zup"]["min"], dtype=float)
    bbox_max = np.asarray(segment["crop_bbox_zup"]["max"], dtype=float)
    ax.add_patch(Rectangle(
        (bbox_min[0], bbox_min[1]), bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1],
        fill=False, edgecolor="black", linewidth=1.3, linestyle=":", label="crop bbox",
    ))
    ax.set_aspect("equal")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_title(
        f"{segment['label']} — z={actual_z:.2f} m, {size}³, "
        f"voxel={float(segment['voxel_m_actual']):.3f} m"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output = plan_dir / segment["artifacts"]["slice_png"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return output


def _print_validation(rows: list[dict[str, Any]]) -> None:
    headers = ("segment", "probe", "point (x,y,z)", "expected", "value_m", "verdict")
    formatted = []
    for row in rows:
        point = row["point_zup"]
        point_text = "-" if point is None else "(" + ",".join(f"{v:.2f}" for v in point) + ")"
        value = "-" if row["value_m"] is None else f"{row['value_m']:+.4f}"
        formatted.append((row["segment"], row["probe"], point_text,
                          row["expected"], value, row["verdict"]))
    widths = [len(header) for header in headers]
    for values in formatted:
        widths = [max(width, len(str(value))) for width, value in zip(widths, values)]
    print("\nValidation")
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for values in formatted:
        print("  ".join(str(value).ljust(width) for value, width in zip(values, widths)))


def validate_plan(plan_path: str | Path) -> bool:
    """Validate every segment in an existing plan, update plan.json, and plot slices."""
    plan_path = Path(plan_path).resolve()
    plan_dir = plan_path.parent
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    try:
        objects = _load_scene_objects(Path(plan["scene_path"]))
        scene_error = None
    except Exception as exc:
        objects = []
        scene_error = str(exc)

    slice_paths: list[str] = []
    for segment in plan["segments"]:
        schema_rows, grid, cfg = _check_schemas(plan_dir, segment)
        rows.extend(schema_rows)
        if grid is None or cfg is None:
            continue
        if scene_error is not None:
            rows.append(_schema_row(segment["seg_id"], "original scene probes", "FAIL", scene_error))
        else:
            try:
                rows.extend(_run_probes(segment, objects, grid, cfg))
            except Exception as exc:
                rows.append(_schema_row(segment["seg_id"], "SDF sign probes", "FAIL", str(exc)))
        try:
            rendered = _render_slice(plan_dir, segment, grid, cfg)
            slice_paths.append(_relative_path(rendered, plan_dir))
            rows.append(_schema_row(segment["seg_id"], "slice PNG", "PASS",
                                    _relative_path(rendered, plan_dir)))
        except Exception as exc:
            rows.append(_schema_row(segment["seg_id"], "slice PNG", "FAIL", str(exc)))

    has_fail = any(row["verdict"] == "FAIL" for row in rows)
    has_warn = any(row["verdict"] == "WARN" for row in rows)
    status = "FAIL" if has_fail else ("PASS_WITH_WARNINGS" if has_warn else "PASS")
    plan["validation"] = {"status": status, "rows": rows, "slice_pngs": slice_paths}
    _write_json(plan_path, plan)
    _print_validation(rows)
    print(f"Validation status: {status}")
    return not has_fail


def compile_affordance(script_path: str | Path, out_root: str | Path = DEFAULT_OUT,
                       volume_root: str = DEFAULT_VOLUME_ROOT,
                       max_grid_override: int | None = None,
                       run_validation: bool = True) -> tuple[Path, bool]:
    """Compile a v0 affordance script and optionally run built-in validation."""
    script_path = Path(script_path).resolve()
    spec = _validate_script(json.loads(script_path.read_text(encoding="utf-8")))
    scene_path = _resolve_repo_path(spec["scene"])
    if not scene_path.is_file():
        raise FileNotFoundError(f"Scene GLB not found: {scene_path}")
    defaults = dict(DEFAULTS)
    defaults.update(spec.get("defaults", {}))
    if max_grid_override is not None:
        if max_grid_override < 64:
            raise ValueError("--max-grid must be >= 64")
        defaults["max_grid"] = int(max_grid_override)

    pose_check = _verify_canonical_pose()
    objects = _load_scene_objects(scene_path)
    plan_name = script_path.stem
    out_root = Path(out_root).resolve()
    plan_dir = out_root / plan_name
    qa_dir = out_root / "qa"
    plan_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    plan: dict[str, Any] = {
        "title": spec["title"],
        "script_path": _path_text(script_path),
        "scene_path": _path_text(scene_path),
        "conventions": {
            "axis_transform": {
                "from": "glTF Y-up",
                "to": "DART/AMASS Z-up",
                "mapping": "(x, y, z) -> (x, -z, y)",
                "matrix_4x4": YUP_TO_ZUP.tolist(),
            },
            "heading": (
                "heading=atan2(dx,dz) in Y-up XZ; 0deg faces +z, 90deg faces +x; "
                "Z-up facing=(sin(theta),-cos(theta)) in XY"
            ),
            "units": "meters",
            "fps": 30,
            "body_model": "SMPL-X male zero-beta",
            "sdf_units": "grid is mesh2sdf-normalized; meters = value/scale",
            "sdf_domain": (
                "cubic grid centered on crop bbox; cube side=max_bbox_extent/0.8 and "
                "therefore extends beyond smaller bbox dimensions"
            ),
            "sdf_recipe": {
                "mesh_scale": 0.8,
                "fix": True,
                "level": "2.0/size",
                "rehearsal_deviation": (
                    "The rehearsal used DART's 1/size level. This compiler uses "
                    "mesh2sdf's documented 2/size level because fix=True keeps only "
                    "the largest connected level-set component, and 1/size can "
                    "fragment thin slabs and flood their interiors positive."
                ),
                "fix_recipe_bias": (
                    "The recomputed SDF measures distance to a level-set surface "
                    "about level outside the true surface, so obstacles read about "
                    "one voxel fatter. The rehearsal's global grid had the same-"
                    "magnitude bias at 3.9 cm."
                ),
            },
            "canonical_pose": {
                "pelvis_z": CANON_PELVIS_Z,
                "left_hip_offset_facing_plus_x": list(CANON_LHIP_OFF),
                "right_hip_offset_facing_plus_x": list(CANON_RHIP_OFF),
                "provenance": (
                    "frame 0 of output/dart_rehearsal/experiment/b_sdf/sample_0.pkl; "
                    "stand.pkl -> SMPL-X male zero-beta; yaw-unrotated -38.65977559deg"
                ),
                "compile_check": pose_check,
            },
        },
        "note": (
            "v0 segments are INDEPENDENT DART runs (per-segment SDF => per-segment "
            "scene_dir). DART multi-interaction history chaining needs a shared scene; "
            "cross-segment stitching is downstream, e.g. choreo.py-style blending."
        ),
        "defaults": {
            "voxel_m": float(defaults["voxel_m"]),
            "corridor_halfwidth_m": float(defaults["corridor_halfwidth_m"]),
            "margin_m": float(defaults["margin_m"]),
            "max_grid": int(defaults["max_grid"]),
        },
        "segments": [],
        "validation": {"status": "NOT_RUN", "rows": [], "slice_pngs": []},
    }

    start_block = dict(spec["start"])
    previous = start_block
    padding = float(defaults["corridor_halfwidth_m"]) + float(defaults["margin_m"])
    for index, station in enumerate(spec["stations"]):
        seg_id = f"seg_{index + 1:02d}_{_slug(station['label'])}"
        start_yup = np.asarray(previous["at"], dtype=float)
        goal_yup = np.asarray(station["at"], dtype=float)
        start_xy = _zup_xy(start_yup)
        goal_xy = _zup_xy(goal_yup)
        distance = float(np.linalg.norm(goal_xy - start_xy))
        facing, facing_source = _segment_facing(previous, start_xy, goal_xy)
        start_surface = 0.0 if index == 0 else float(previous.get("surface_z", 0.0))
        goal_surface = float(station.get("surface_z", 0.0))
        init_joints, goal_joints = _standing_joints(
            start_xy, facing, start_surface, goal_xy, goal_surface)
        prompt = _text_prompt(station["text"], distance)
        bbox_min, bbox_max = _crop_bbox(objects, start_xy, goal_xy, padding)
        cropped_mesh, segment_warnings = _crop_scene_mesh(objects, bbox_min, bbox_max)
        max_extent = float((bbox_max - bbox_min).max())
        size, needed = _grid_size(max_extent, float(defaults["voxel_m"]),
                                  int(defaults["max_grid"]))
        if _next_power_of_two(needed) > int(defaults["max_grid"]):
            segment_warnings.append(
                f"voxel target needs >= {_next_power_of_two(needed)} grid; "
                f"clamped to max_grid={int(defaults['max_grid'])}"
            )

        seg_dir = plan_dir / seg_id
        scene_dir = seg_dir / "scene"
        seg_dir.mkdir(parents=True, exist_ok=True)
        crop_npz = seg_dir / "sdf_crop.npz"
        _cfg, voxel_actual = _make_sdf(
            cropped_mesh, bbox_min, bbox_max, size, scene_dir, crop_npz)
        volume_root_posix = volume_root.replace("\\", "/")
        volume_scene_dir = str(
            PurePosixPath(volume_root_posix) / plan_name / seg_id)
        volume_path_cli_root = (
            "/" + volume_root_posix.removeprefix("/vol/")
            if volume_root_posix.startswith("/vol/")
            else volume_root_posix
        )
        volume_path_cli = str(
            PurePosixPath(volume_path_cli_root) / plan_name / seg_id)
        interaction = {
            "scene_dir": volume_scene_dir,
            "floor_height": 0.0,
            "interaction_name": seg_id,
            "interactions": [{
                "text_prompt": prompt,
                "init_joints": init_joints,
                "goal_joints": goal_joints,
            }],
        }
        interaction_path = seg_dir / "interaction.json"
        _write_json(interaction_path, interaction)
        slice_path = qa_dir / f"{plan_name}_{seg_id}_slice.png"
        artifacts = {
            "scene_sdf_npy": _relative_path(scene_dir / "scene_sdf.npy", plan_dir),
            "scene_sdf_json": _relative_path(scene_dir / "scene_sdf.json", plan_dir),
            "scene_with_floor_obj": _relative_path(scene_dir / "scene_with_floor.obj", plan_dir),
            "interaction_json": _relative_path(interaction_path, plan_dir),
            "sdf_crop_npz": _relative_path(crop_npz, plan_dir),
            "slice_png": _relative_path(slice_path, plan_dir),
        }
        local_scene_text = scene_dir.resolve().as_posix()
        segment: dict[str, Any] = {
            "seg_id": seg_id,
            "label": station["label"],
            "text_prompt": prompt,
            "distance_xy_m": distance,
            "start_yup_xz": start_yup.tolist(),
            "goal_yup_xz": goal_yup.tolist(),
            "start_zup_xy": start_xy.tolist(),
            "goal_zup_xy": goal_xy.tolist(),
            "surface_z": {"start": start_surface, "goal": goal_surface},
            "init_facing_zup_xy": facing.tolist(),
            "init_facing_source": facing_source,
            "init_joints": init_joints,
            "goal_joints": goal_joints,
            "crop_bbox_zup": {"min": bbox_min.tolist(), "max": bbox_max.tolist()},
            "grid_size": size,
            "voxel_m_actual": voxel_actual,
            "warnings": segment_warnings,
            "artifacts": artifacts,
            "volume_scene_dir": volume_scene_dir,
            "volume_path_cli": volume_path_cli,
            "upload_command": (
                f'modal volume put dart-assets "{local_scene_text}" "{volume_path_cli}"'
            ),
        }
        for metadata_key in ("target_object", "contact_ref"):
            if metadata_key in station:
                segment[metadata_key] = station[metadata_key]
        plan["segments"].append(segment)
        print(
            f"[{seg_id}] bbox min={bbox_min.round(3).tolist()} "
            f"max={bbox_max.round(3).tolist()} | grid={size}³ | "
            f"voxel={voxel_actual:.4f} m"
        )
        print(f"  scene: {_relative_path(scene_dir, plan_dir)}")
        print(f"  interaction: {artifacts['interaction_json']} | crop: {artifacts['sdf_crop_npz']}")
        previous = station

    plan_path = plan_dir / "plan.json"
    _write_json(plan_path, plan)
    print(f"Plan: {plan_path}")
    if not run_validation:
        return plan_path, True
    return plan_path, validate_plan(plan_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile", help="compile an affordance script")
    compile_parser.add_argument("--script", required=True, type=Path)
    compile_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    compile_parser.add_argument("--volume-root", default=DEFAULT_VOLUME_ROOT)
    compile_parser.add_argument("--max-grid", type=int, default=None)
    compile_parser.add_argument("--no-validate", action="store_true")
    validate_parser = subparsers.add_parser("validate", help="validate an existing plan")
    validate_parser.add_argument("--plan", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "compile":
            _plan_path, valid = compile_affordance(
                args.script,
                out_root=args.out,
                volume_root=args.volume_root,
                max_grid_override=args.max_grid,
                run_validation=not args.no_validate,
            )
        else:
            valid = validate_plan(args.plan)
        return 0 if valid else 1
    except (OSError, ValueError, TypeError, AssertionError, RuntimeError) as exc:
        print(f"affordance_compile: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
