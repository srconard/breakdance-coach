"""Prepare a scanned environment for the move-study viewer.

This is the normal Python entry point. It locates Blender and delegates the
mesh work to ``blender_scripts/prepare_scene_glb.py``, which imports,
decimates, grounds, exports, and re-imports the scene for validation.

Example:
    python -m analyzer_3d.src.prepare_scene scan.ply \
        -o output/scenes/scan.glb --target-triangles 200000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BLENDER_SCRIPT = Path(__file__).with_name("blender_scripts") / "prepare_scene_glb.py"
WINDOWS_BLENDER_CANDIDATES = (
    Path(r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"),
)


def find_blender(explicit: str | None = None) -> Path:
    """Return a usable Blender executable or raise a concise error."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Blender executable not found: {candidate}")

    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path).resolve()

    for candidate in WINDOWS_BLENDER_CANDIDATES:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Blender was not found on PATH or in a known Windows install location. "
        "Pass --blender PATH."
    )


def prepare_scene(
    source: str | Path,
    output: str | Path,
    *,
    blender: str | None = None,
    target_triangles: int = 200_000,
    max_output_mb: float = 25.0,
    max_face_area: float = 0.25,
    scale: float = 1.0,
    level_mode: str = "pca",
    center_mode: str = "walkable",
    ground_mode: str = "local",
    ground_radius: float = 2.0,
    ground_percentile: float = 10.0,
    walkable_search_radius: float = 12.0,
    walkable_max_slope: float = 20.0,
    walkable_min_area: float = 1.2,
    origin_x: float | None = None,
    origin_y: float | None = None,
    ground_z: float | None = None,
    report: str | Path | None = None,
) -> dict:
    """Run Blender scene preparation and return its validation report."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Scene source not found: {source_path}")

    output_path = Path(output).expanduser().resolve()
    if output_path == source_path:
        raise ValueError("Source and output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(report).expanduser().resolve()
        if report
        else output_path.with_suffix(".scene-report.json")
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # Blender can exit 0 even when a Python script raises. Removing an older
    # report prevents a failed run from being mistaken for fresh validation.
    report_path.unlink(missing_ok=True)

    command = [
        str(find_blender(blender)),
        "--background",
        "--factory-startup",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        str(source_path),
        str(output_path),
        "--report",
        str(report_path),
        "--target-triangles",
        str(target_triangles),
        "--max-output-mb",
        str(max_output_mb),
        "--max-face-area",
        str(max_face_area),
        "--scale",
        str(scale),
        "--level-mode",
        level_mode,
        "--center-mode",
        center_mode,
        "--ground-mode",
        ground_mode,
        "--ground-radius",
        str(ground_radius),
        "--ground-percentile",
        str(ground_percentile),
        "--walkable-search-radius",
        str(walkable_search_radius),
        "--walkable-max-slope",
        str(walkable_max_slope),
        "--walkable-min-area",
        str(walkable_min_area),
    ]
    for name, value in (
        ("--origin-x", origin_x),
        ("--origin-y", origin_y),
        ("--ground-z", ground_z),
    ):
        if value is not None:
            command.extend((name, str(value)))

    print("[prepare_scene] running Blender scene preparation...")
    subprocess.run(command, cwd=REPO, check=True)

    if not report_path.is_file():
        raise ValueError("Blender did not produce a validation report")

    result = json.loads(report_path.read_text(encoding="utf-8"))
    if result["output"].get("failed_checks"):
        raise ValueError(
            "GLB validation failed: "
            + ", ".join(result["output"]["failed_checks"])
        )
    print(
        "[prepare_scene] "
        f"{result['output']['triangles']:,} triangles, "
        f"{result['output']['size_mb']:.1f} MB, "
        f"ground error {result['output']['ground_error_m']:.4f} m"
    )
    print(f"[prepare_scene] report: {report_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="source PLY, OBJ, FBX, GLB, or glTF mesh")
    parser.add_argument("-o", "--output", required=True, help="output GLB path")
    parser.add_argument("--blender", help="Blender executable (auto-detected by default)")
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=200_000,
        help="decimation ceiling (default: 200000)",
    )
    parser.add_argument(
        "--max-output-mb",
        type=float,
        default=25.0,
        help="fail validation above this GLB size (default: 25)",
    )
    parser.add_argument(
        "--max-face-area",
        type=float,
        default=0.25,
        help="remove post-decimation hole-fill faces larger than this many m^2",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="source-units to meters multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--level-mode",
        choices=("pca", "none"),
        default="pca",
        help="auto-level a sheet-like scan with PCA, or preserve source axes",
    )
    parser.add_argument(
        "--center-mode",
        choices=("walkable", "bounds", "none"),
        default="walkable",
        help="put a nearby walkable patch, bounds center, or source origin at world origin",
    )
    parser.add_argument(
        "--ground-mode",
        choices=("local", "min", "none"),
        default="local",
        help="ground from the local dance spot, global minimum, or no shift",
    )
    parser.add_argument(
        "--ground-radius",
        type=float,
        default=2.0,
        help="radius around the origin used by local grounding (default: 2 m)",
    )
    parser.add_argument(
        "--ground-percentile",
        type=float,
        default=10.0,
        help="local height percentile treated as ground (default: 10)",
    )
    parser.add_argument(
        "--walkable-search-radius",
        type=float,
        default=12.0,
        help="search radius around bounds center for a support patch (default: 12 m)",
    )
    parser.add_argument(
        "--walkable-max-slope",
        type=float,
        default=20.0,
        help="maximum support-patch slope in degrees (default: 20)",
    )
    parser.add_argument(
        "--walkable-min-area",
        type=float,
        default=1.2,
        help="minimum coherent horizontal area for the dancer (default: 1.2 m^2)",
    )
    parser.add_argument("--origin-x", type=float, help="source X coordinate for world X=0")
    parser.add_argument("--origin-y", type=float, help="source Y coordinate for world Y=0")
    parser.add_argument("--ground-z", type=float, help="source Z coordinate for ground=0")
    parser.add_argument("--report", help="validation JSON path")
    args = parser.parse_args()

    prepare_scene(
        args.source,
        args.output,
        blender=args.blender,
        target_triangles=args.target_triangles,
        max_output_mb=args.max_output_mb,
        max_face_area=args.max_face_area,
        scale=args.scale,
        level_mode=args.level_mode,
        center_mode=args.center_mode,
        ground_mode=args.ground_mode,
        ground_radius=args.ground_radius,
        ground_percentile=args.ground_percentile,
        walkable_search_radius=args.walkable_search_radius,
        walkable_max_slope=args.walkable_max_slope,
        walkable_min_area=args.walkable_min_area,
        origin_x=args.origin_x,
        origin_y=args.origin_y,
        ground_z=args.ground_z,
        report=args.report,
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"[prepare_scene] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
