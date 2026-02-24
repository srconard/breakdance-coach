"""GLB exporter — converts SMPL parameters to animated .GLB files.

Supports two modes:
  - Local: Calls Blender headless on the local machine
  - (Future) Modal: Runs Blender headless on Modal cloud

Usage:
    from analyzer_3d.src.exporter import export_glb
    export_glb("pose_output.pkl", "animation.glb")
"""

import shutil
import subprocess
import sys
from pathlib import Path


# Path to the Blender script (relative to this file)
BLENDER_SCRIPT = Path(__file__).parent / "blender_scripts" / "smpl_to_glb.py"


def find_blender() -> str | None:
    """Find Blender executable on the system."""
    # Check if 'blender' is in PATH
    blender_path = shutil.which("blender")
    if blender_path:
        return blender_path

    # Common installation paths on Windows
    blender_dir = Path(r"C:\Program Files\Blender Foundation")
    if blender_dir.exists():
        # Find the newest Blender version installed
        versions = sorted(blender_dir.glob("Blender */blender.exe"), reverse=True)
        if versions:
            return str(versions[0])

    return None


def export_glb(
    input_pkl: str | Path,
    output_glb: str | Path | None = None,
    fps: int | None = None,
    blender_path: str | None = None,
) -> Path:
    """Convert SMPL parameters (pickle) to animated GLB using Blender.

    Args:
        input_pkl: Path to pickle file with SMPL parameters (from GVHMR)
        output_glb: Path for output GLB file (auto-generated if None)
        fps: Override frame rate (default: use FPS from pickle)
        blender_path: Path to Blender executable (auto-detected if None)

    Returns:
        Path to the created GLB file

    Raises:
        FileNotFoundError: If Blender or input file not found
        RuntimeError: If Blender export fails
    """
    input_pkl = Path(input_pkl)
    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pickle not found: {input_pkl}")

    if output_glb is None:
        output_glb = input_pkl.with_suffix(".glb")
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    # Find Blender
    if blender_path is None:
        blender_path = find_blender()
    if blender_path is None:
        raise FileNotFoundError(
            "Blender not found. Install Blender 4.x and ensure it's in PATH,\n"
            "or specify the path with --blender-path."
        )

    print(f"GLB Export (Blender Headless)")
    print(f"  Input:   {input_pkl}")
    print(f"  Output:  {output_glb}")
    print(f"  Blender: {blender_path}")

    # Build command
    cmd = [
        blender_path,
        "-b",  # Background/headless
        "--python", str(BLENDER_SCRIPT),
        "--",
        "--input", str(input_pkl.resolve()),
        "--output", str(output_glb.resolve()),
    ]
    if fps is not None:
        cmd.extend(["--fps", str(fps)])

    # Run Blender
    print("  Running Blender headless...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        # Print Blender output (filtered to our script's prints)
        for line in result.stdout.splitlines():
            if any(kw in line for kw in ["SMPL", "Creating", "Applying", "Exporting", "Done", "Keyframed", "frames"]):
                print(f"  {line}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else e.stdout or str(e)
        raise RuntimeError(f"Blender export failed:\n{error_msg}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Blender not found at: {blender_path}")

    if not output_glb.exists():
        raise RuntimeError(f"GLB file was not created: {output_glb}")

    size_mb = output_glb.stat().st_size / (1024 * 1024)
    print(f"  Created: {output_glb} ({size_mb:.2f} MB)")

    return output_glb


def convert_fbx_to_glb(
    input_fbx: str | Path,
    output_glb: str | Path | None = None,
    blender_path: str | None = None,
) -> Path:
    """Convert FBX (e.g. from DeepMotion) to GLB using Blender.

    Args:
        input_fbx: Path to FBX file
        output_glb: Path for output GLB (auto-generated if None)
        blender_path: Path to Blender (auto-detected if None)

    Returns:
        Path to the created GLB file
    """
    input_fbx = Path(input_fbx)
    if not input_fbx.exists():
        raise FileNotFoundError(f"Input FBX not found: {input_fbx}")

    if output_glb is None:
        output_glb = input_fbx.with_suffix(".glb")
    output_glb = Path(output_glb)
    output_glb.parent.mkdir(parents=True, exist_ok=True)

    if blender_path is None:
        blender_path = find_blender()
    if blender_path is None:
        raise FileNotFoundError("Blender not found.")

    print(f"FBX to GLB Conversion")
    print(f"  Input:  {input_fbx}")
    print(f"  Output: {output_glb}")

    # Inline Blender script for FBX → GLB conversion
    script = f"""
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=r"{input_fbx.resolve()}")
bpy.ops.export_scene.gltf(
    filepath=r"{output_glb.resolve()}",
    export_format='GLB',
    export_animations=True,
)
"""

    cmd = [blender_path, "-b", "--python-expr", script]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FBX to GLB conversion failed:\n{e.stderr or e.stdout}")

    if not output_glb.exists():
        raise RuntimeError(f"GLB file was not created: {output_glb}")

    size_mb = output_glb.stat().st_size / (1024 * 1024)
    print(f"  Created: {output_glb} ({size_mb:.2f} MB)")

    return output_glb
