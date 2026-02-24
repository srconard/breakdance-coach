"""Setup script for GVHMR checkpoints on Modal.

Downloads model checkpoints and uploads them to a Modal Volume so
the GVHMR inference container can access them.

Prerequisites:
    1. Register at smpl.is.tue.mpg.de → download SMPL for Python
    2. Register at smpl-x.is.tue.mpg.de → download SMPL-X v1.1
    3. Place files in local_checkpoints/ directory (see structure below)
    4. Run: python -m analyzer_3d.src.gvhmr_setup

Expected local_checkpoints/ structure:
    local_checkpoints/
    ├── body_models/
    │   ├── smpl/
    │   │   └── SMPL_NEUTRAL.pkl
    │   └── smplx/
    │       └── SMPLX_NEUTRAL.npz
    └── (other checkpoints downloaded automatically)

The script will:
    1. Download GVHMR, HMR2, ViTPose, and YOLOv8 checkpoints from Hugging Face
    2. Upload everything (including your SMPL files) to Modal Volume
"""

import os
import sys
import urllib.request
from pathlib import Path


# Hugging Face mirror of GVHMR checkpoints (camenduru)
# More reliable than Google Drive for automated downloads
CHECKPOINT_URLS = {
    "gvhmr/gvhmr_siga24_release.ckpt":
        "https://huggingface.co/camenduru/GVHMR/resolve/main/gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt":
        "https://huggingface.co/camenduru/GVHMR/resolve/main/hmr2/epoch%3D10-step%3D25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth":
        "https://huggingface.co/camenduru/GVHMR/resolve/main/vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt":
        "https://huggingface.co/camenduru/GVHMR/resolve/main/yolo/yolov8x.pt",
}

LOCAL_CHECKPOINT_DIR = Path("local_checkpoints")


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Already exists: {dest.name} ({size_mb:.1f} MB)")
        return

    print(f"  Downloading {desc or dest.name}...")

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size * 100 / total_size, 100)
            mb_done = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    {pct:.0f}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"\r    Done ({size_mb:.1f} MB)              ")
    except Exception as e:
        print(f"\r    FAILED: {e}")
        if dest.exists():
            dest.unlink()
        raise


def download_checkpoints():
    """Download all required checkpoints to local directory."""
    print("\n=== Downloading Checkpoints (from Hugging Face) ===\n")

    for rel_path, url in CHECKPOINT_URLS.items():
        dest = LOCAL_CHECKPOINT_DIR / rel_path
        download_file(url, dest, desc=rel_path)


def check_smpl_files():
    """Check that SMPL body model files are present."""
    print("\n=== Checking SMPL Body Models ===\n")

    smpl_file = LOCAL_CHECKPOINT_DIR / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"
    smplx_file = LOCAL_CHECKPOINT_DIR / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"

    missing = []
    if not smpl_file.exists():
        missing.append(str(smpl_file))
    else:
        size_mb = smpl_file.stat().st_size / (1024 * 1024)
        print(f"  Found: {smpl_file.name} ({size_mb:.1f} MB)")

    if not smplx_file.exists():
        missing.append(str(smplx_file))
    else:
        size_mb = smplx_file.stat().st_size / (1024 * 1024)
        print(f"  Found: {smplx_file.name} ({size_mb:.1f} MB)")

    if missing:
        print("\n  MISSING SMPL files:")
        for f in missing:
            print(f"    - {f}")
        print(
            "\n  Please download from:"
            "\n    SMPL:  https://smpl.is.tue.mpg.de/"
            "\n    SMPLX: https://smpl-x.is.tue.mpg.de/"
            f"\n\n  Place them in: {LOCAL_CHECKPOINT_DIR}/body_models/"
        )
        return False

    return True


def upload_to_modal():
    """Upload all checkpoints to Modal Volume."""
    import modal
    from pathlib import PurePosixPath

    print("\n=== Uploading to Modal Volume ===\n")

    vol = modal.Volume.from_name("gvhmr-checkpoints", create_if_missing=True)

    with vol.batch_upload(force=True) as batch:
        for filepath in sorted(LOCAL_CHECKPOINT_DIR.rglob("*")):
            if filepath.is_file():
                rel_path = filepath.relative_to(LOCAL_CHECKPOINT_DIR)
                remote_path = "/" + PurePosixPath(rel_path).as_posix()
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"  Uploading {rel_path} ({size_mb:.1f} MB) -> {remote_path}")
                batch.put_file(str(filepath), remote_path)

    print("\n  All checkpoints uploaded to Modal volume 'gvhmr-checkpoints'!")


def verify_volume():
    """Verify the Modal volume has all required files."""
    import modal

    print("\n=== Verifying Modal Volume ===\n")

    vol = modal.Volume.from_name("gvhmr-checkpoints")

    required = [
        "gvhmr/gvhmr_siga24_release.ckpt",
        "hmr2/epoch=10-step=25000.ckpt",
        "vitpose/vitpose-h-multi-coco.pth",
        "yolo/yolov8x.pt",
        "body_models/smplx/SMPLX_NEUTRAL.npz",
        "body_models/smpl/SMPL_NEUTRAL.pkl",
    ]

    all_ok = True
    for path in required:
        try:
            entries = vol.listdir(f"/{path.rsplit('/', 1)[0]}")
            filename = path.rsplit("/", 1)[1]
            found = any(e.path.endswith(filename) for e in entries)
            if found:
                print(f"  OK: {path}")
            else:
                print(f"  MISSING: {path}")
                all_ok = False
        except Exception:
            print(f"  MISSING: {path}")
            all_ok = False

    if all_ok:
        print("\n  All checkpoints verified!")
    else:
        print("\n  Some checkpoints are missing. Re-run setup.")

    return all_ok


def main():
    print("=" * 55)
    print("GVHMR CHECKPOINT SETUP FOR MODAL")
    print("=" * 55)

    # Step 1: Download checkpoints
    download_checkpoints()

    # Step 2: Check SMPL files
    if not check_smpl_files():
        print("\nCannot proceed without SMPL body model files.")
        print("Please download them and re-run this script.")
        return 1

    # Step 3: Upload to Modal
    upload_to_modal()

    # Step 4: Verify
    verify_volume()

    print("\n" + "=" * 55)
    print("SETUP COMPLETE!")
    print("=" * 55)
    print("\nNext steps:")
    print("  1. Deploy GVHMR: python -m modal deploy analyzer_3d/src/gvhmr_modal.py")
    print("  2. Test: python -m analyzer_3d.src.main video.mp4 --backend gvhmr")

    return 0


if __name__ == "__main__":
    sys.exit(main())
