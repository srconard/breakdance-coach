"""Retarget a DART SMPL-X sample into a mannequin/viewer bundle.

DART exports the first 22 SMPL-X joints as measured Z-up world positions
plus local rotation matrices at 30 fps. This bridge derives a compatible
SMPL-schema rest skeleton from those positions, extends it with the two SMPL
hand joints, converts the motion to the repository's Y-up convention, and
optionally upsamples it to 60 fps before invoking the standard mannequin GLB
builder and self-contained viewer packer.

Usage:
    python -m analyzer_3d.src.dart_retarget sample_0.pkl
    python -m analyzer_3d.src.dart_retarget sample_0.pkl --no-render
        [-o output/dart_derisk/retarget] [--keep-fps]
"""

import argparse
import json
import pathlib
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from .make_viewer import make_viewer
from .mannequin import glb_from_npz
from .motion_cleanup import (
    EFFECTORS,
    HEIGHT_THRESH,
    PARENTS24,
    SPEED_THRESH,
    detect_contacts,
    fk,
)


SOURCE_FPS = 30.0
DEFAULT_OUT_DIR = Path("output/dart_derisk/retarget")
DEFAULT_BASELINE = Path(
    "output/reference_library/aistpp/"
    "gBR_sBM_cAll_d04_mBR0_ch01_smpl_clean.pkl"
)
HAND_LENGTH_M = 0.0862

# DART/AMASS Z-up -> repository Y-up: (x, y, z) -> (x, z, -y).
ZUP_TO_YUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
FOOT_EFFECTORS = ("L_ankle", "R_ankle")


def _as_float64(value):
    """Move a torch tensor to CPU numpy, or coerce an array, as float64."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _load_dart_pkl(path):
    """Load a DART pickle, including Linux PosixPath objects on Windows."""
    path = Path(path)
    original_posix_path = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    finally:
        pathlib.PosixPath = original_posix_path

    # DART stores every numeric payload as a CPU torch tensor. Convert all of
    # them here so no torch objects leak into the retargeting calculations.
    return {
        key: _as_float64(value) if hasattr(value, "detach") else value
        for key, value in data.items()
    }


def _validate_dart_arrays(data):
    required = {"joints", "global_orient", "body_pose", "transl"}
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"DART sample is missing required keys: {', '.join(missing)}")

    joints = _as_float64(data["joints"])
    global_orient = _as_float64(data["global_orient"])
    body_pose = _as_float64(data["body_pose"])
    transl = _as_float64(data["transl"])
    frames = joints.shape[0] if joints.ndim else 0

    expected = {
        "joints": (frames, 22, 3),
        "global_orient": (frames, 3, 3),
        "body_pose": (frames, 21, 3, 3),
        "transl": (frames, 3),
    }
    actual = {
        "joints": joints.shape,
        "global_orient": global_orient.shape,
        "body_pose": body_pose.shape,
        "transl": transl.shape,
    }
    for name, expected_shape in expected.items():
        if actual[name] != expected_shape:
            raise ValueError(
                f"Expected {name} shape {expected_shape}, got {actual[name]}"
            )
    if frames == 0:
        raise ValueError("DART sample contains no frames")
    for name, array in (
        ("joints", joints),
        ("global_orient", global_orient),
        ("body_pose", body_pose),
        ("transl", transl),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"DART sample contains non-finite values in {name}")
    return joints, global_orient, body_pose


def _chain_world_rotations(global_orient, body_pose):
    """Build the first 22 SMPL-X world rotations from local rotations."""
    frames = global_orient.shape[0]
    world = np.empty((frames, 22, 3, 3), dtype=np.float64)
    world[:, 0] = global_orient
    for joint in range(1, 22):
        parent = PARENTS24[joint]
        world[:, joint] = world[:, parent] @ body_pose[:, joint - 1]
    return world


def _derive_rest_skeleton(joints, world_rotations):
    """Estimate parent-frame rest offsets and accumulate 22 rest joints."""
    rest = np.zeros((22, 3), dtype=np.float64)
    mean_offsets = np.zeros((22, 3), dtype=np.float64)
    max_component_std = 0.0

    for child in range(1, 22):
        parent = PARENTS24[child]
        delta_world = joints[:, child] - joints[:, parent]
        offsets = np.einsum(
            "fji,fj->fi", world_rotations[:, parent], delta_world
        )
        mean_offsets[child] = offsets.mean(axis=0)
        max_component_std = max(
            max_component_std, float(offsets.std(axis=0).max())
        )
        rest[child] = rest[parent] + mean_offsets[child]

    return rest, {
        "fk_consistency_max_std_m": max_component_std,
    }


def _extend_to_smpl24(joints22, rotations22, rest22):
    """Continue each forearm with a fixed-length SMPL hand endpoint."""
    frames = joints22.shape[0]
    joints = np.empty((frames, 24, 3), dtype=np.float64)
    rotations = np.empty((frames, 24, 3, 3), dtype=np.float64)
    rest = np.empty((24, 3), dtype=np.float64)
    joints[:, :22] = joints22
    rotations[:, :22] = rotations22
    rest[:22] = rest22

    for hand, wrist, elbow in ((22, 20, 18), (23, 21, 19)):
        forearm = rest[wrist] - rest[elbow]
        length = float(np.linalg.norm(forearm))
        if length < 1e-12:
            raise ValueError(f"Cannot extend zero-length forearm for joint {hand}")
        rest[hand] = rest[wrist] + HAND_LENGTH_M * forearm / length
        rotations[:, hand] = rotations[:, wrist]
        hand_offset = rest[hand] - rest[wrist]
        joints[:, hand] = joints[:, wrist] + np.einsum(
            "fij,j->fi", rotations[:, wrist], hand_offset
        )
    return joints, rotations, rest


def _to_y_up(joints, rotations, rest):
    joints_yup = joints @ ZUP_TO_YUP.T
    rotations_yup = (
        ZUP_TO_YUP[None, None] @ rotations @ ZUP_TO_YUP.T[None, None]
    )
    rest_yup = rest @ ZUP_TO_YUP.T
    return joints_yup, rotations_yup, rest_yup


def _upsample_2x(joints, rotations):
    """Upsample positions and rotations at exact temporal half-steps."""
    frames = joints.shape[0]
    if frames == 1:
        return joints.copy(), rotations.copy()

    source_times = np.arange(frames, dtype=np.float64)
    output_times = np.arange(2 * frames - 1, dtype=np.float64) * 0.5
    joints_out = CubicSpline(source_times, joints, axis=0)(output_times)
    rotations_out = np.empty(
        (output_times.size, rotations.shape[1], 3, 3), dtype=np.float64
    )

    for joint in range(rotations.shape[1]):
        quaternions = Rotation.from_matrix(rotations[:, joint]).as_quat()
        for frame in range(1, frames):
            if np.dot(quaternions[frame - 1], quaternions[frame]) < 0.0:
                quaternions[frame] *= -1.0
        rotations_out[:, joint] = Slerp(
            source_times, Rotation.from_quat(quaternions)
        )(output_times).as_matrix()
    return joints_out, rotations_out


def _toe_metrics(joints_world):
    toe_heights = joints_world[:, (10, 11), 1]
    return {
        "toe_height_min_m": float(toe_heights.min()),
        "toe_height_mean_m": float(toe_heights.mean()),
    }


def _segment_endpoint_error(joints_world, world_rotations, rest):
    """Mean error between supplied child positions and rigid bone endpoints."""
    endpoint_errors = []
    for child in range(1, 24):
        parent = PARENTS24[child]
        offset = rest[child] - rest[parent]
        reconstructed = joints_world[:, parent] + np.einsum(
            "fij,j->fi", world_rotations[:, parent], offset
        )
        endpoint_errors.append(
            np.linalg.norm(reconstructed - joints_world[:, child], axis=1)
        )
    return float(np.concatenate(endpoint_errors).mean())


def _foot_contact_metrics(joints_world, fps, floor_y=0.0):
    """Measure ankle slide per detected foot-contact segment."""
    contacts = detect_contacts(joints_world, floor_y=floor_y, fps=fps)
    segment_speeds = []
    contact_toe_heights = []
    n_segments = 0
    n_contact_frames = 0

    for name in FOOT_EFFECTORS:
        joint = EFFECTORS[name]["joint"]
        toe = EFFECTORS[name]["tip"]
        for start, end in contacts[name]:
            n_segments += 1
            n_contact_frames += end - start
            contact_toe_heights.extend(joints_world[start:end, toe, 1].tolist())
            if end - start >= 2:
                ankle_xz = joints_world[start:end, joint][:, (0, 2)]
                speed = np.linalg.norm(np.diff(ankle_xz, axis=0), axis=1) * fps
                segment_speeds.append(float(speed.mean()))

    return {
        "contact_slide_cm_s": (
            float(np.mean(segment_speeds) * 100.0) if segment_speeds else 0.0
        ),
        "n_segments": n_segments,
        "n_contact_frames": n_contact_frames,
        "contact_toe_height_mean_m": (
            float(np.mean(contact_toe_heights))
            if contact_toe_heights
            else None
        ),
    }


def _baseline_metrics(path):
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)

    params = data["smpl_params_global"]
    rest = _as_float64(data["mesh"]["joints"])
    fps = float(data["fps"])
    joints_world, _, _ = fk(
        _as_float64(params["body_pose"]),
        _as_float64(params["global_orient"]),
        _as_float64(params["transl"]),
        rest,
    )
    metrics = {
        "path": str(path),
        "fps": fps,
        "n_frames": int(joints_world.shape[0]),
        **_toe_metrics(joints_world),
        **_foot_contact_metrics(joints_world, fps=fps, floor_y=0.0),
    }
    metrics["stored_cleanup_contact_slide_cm_s"] = data.get("cleanup", {}).get(
        "metrics_after", {}
    ).get("contact_slide_cm_s")
    return metrics


def _format_metric(value, precision=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{precision}f}"


def _print_metrics(input_path, frames_in, metrics):
    dart = metrics["dart"]
    baseline = metrics["baseline"]
    print(
        f"DART retarget: {Path(input_path).name} "
        f"({frames_in} frames @ {SOURCE_FPS:g} fps -> "
        f"{dart['n_frames']} frames @ {dart['fps']:g} fps)"
    )
    print()
    print(f"  {'Metric':<38} {'DART':>12} {'Baseline':>12}")
    print(f"  {'-' * 38} {'-' * 12} {'-' * 12}")
    rows = (
        ("contact slide (cm/s)", "contact_slide_cm_s", 2),
        ("stored cleanup slide (cm/s)", "stored_cleanup_contact_slide_cm_s", 2),
        ("foot contact segments", "n_segments", 0),
        ("foot contact frames", "n_contact_frames", 0),
        ("contact toe height mean (m)", "contact_toe_height_mean_m", 4),
        ("toe height min (m)", "toe_height_min_m", 4),
        ("toe height mean (m)", "toe_height_mean_m", 4),
        ("FK consistency max std (m)", "fk_consistency_max_std_m", 6),
        ("segment endpoint error mean (m)", "segment_endpoint_error_mean_m", 6),
    )
    for label, key, precision in rows:
        print(
            f"  {label:<38} "
            f"{_format_metric(dart.get(key), precision):>12} "
            f"{_format_metric(baseline.get(key), precision):>12}"
        )
    print()


def retarget_dart(
    input_pkl,
    out_dir=DEFAULT_OUT_DIR,
    title=None,
    baseline=DEFAULT_BASELINE,
    render=True,
    keep_fps=False,
    blender_path=None,
):
    """Retarget one DART sample and return its generated artifact paths."""
    input_pkl = Path(input_pkl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_pkl.stem
    title = title or f"{stem} (DART retarget)"

    data = _load_dart_pkl(input_pkl)
    joints22, global_orient, body_pose = _validate_dart_arrays(data)
    rotations22 = _chain_world_rotations(global_orient, body_pose)
    rest22, fit_metrics = _derive_rest_skeleton(joints22, rotations22)
    joints, rotations, rest = _extend_to_smpl24(joints22, rotations22, rest22)
    joints, rotations, rest = _to_y_up(joints, rotations, rest)

    if keep_fps:
        out_fps = SOURCE_FPS
    else:
        joints, rotations = _upsample_2x(joints, rotations)
        out_fps = SOURCE_FPS * 2.0

    motion_npz = out_dir / f"{stem}_motion.npz"
    metrics_json = out_dir / f"{stem}_retarget_metrics.json"
    mannequin_glb = out_dir / f"{stem}_mannequin.glb"
    viewer_html = out_dir / f"{stem}_viewer.html"
    np.savez(
        motion_npz,
        JW=joints,
        G=rotations,
        rest=rest,
        fps=np.float64(out_fps),
    )

    dart_metrics = {
        "fps": out_fps,
        "n_frames": int(joints.shape[0]),
        **fit_metrics,
        "segment_endpoint_error_mean_m": _segment_endpoint_error(
            joints, rotations, rest
        ),
        **_toe_metrics(joints),
        **_foot_contact_metrics(joints, fps=out_fps, floor_y=0.0),
    }
    metrics = {
        "input": str(input_pkl),
        "source_fps": SOURCE_FPS,
        "source_frames": int(joints22.shape[0]),
        "upsampled_2x": not keep_fps,
        "contact_detection": {
            "floor_y": 0.0,
            "speed_threshold_m_s": SPEED_THRESH,
            "height_threshold_m": {
                name: HEIGHT_THRESH[name] for name in FOOT_EFFECTORS
            },
        },
        "artifacts": {
            "motion_npz": str(motion_npz),
            "mannequin_glb": str(mannequin_glb) if render else None,
            "viewer_html": str(viewer_html) if render else None,
        },
        "dart": dart_metrics,
        "baseline": _baseline_metrics(baseline),
    }
    metrics_json.write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    _print_metrics(input_pkl, joints22.shape[0], metrics)
    print(f"  Motion NPZ: {motion_npz}")
    print(f"  Metrics:    {metrics_json}")

    result = {
        "motion_npz": str(motion_npz),
        "metrics_json": str(metrics_json),
    }
    if render:
        glb = glb_from_npz(motion_npz, mannequin_glb, blender_path)
        html = make_viewer(
            glb,
            video=None,
            title=title,
            fps=out_fps,
            out=viewer_html,
        )
        result.update({"mannequin_glb": str(glb), "viewer_html": str(html)})
    else:
        print("  Render:     skipped (--no-render)")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="DART sample pickle")
    parser.add_argument(
        "-o",
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--title", default=None, help="viewer title")
    parser.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="clean SMPL library pickle used for foot-skate comparison",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="write the motion NPZ and metrics without running Blender/viewer",
    )
    parser.add_argument(
        "--keep-fps",
        action="store_true",
        help="keep the DART motion at 30 fps instead of upsampling to 60 fps",
    )
    parser.add_argument("--blender-path", default=None)
    args = parser.parse_args()

    retarget_dart(
        args.input,
        out_dir=args.out_dir,
        title=args.title,
        baseline=args.baseline,
        render=not args.no_render,
        keep_fps=args.keep_fps,
        blender_path=args.blender_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
