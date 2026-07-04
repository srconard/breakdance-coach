"""Setup script for WHAM checkpoints on Modal (mocap model spike, 2026-07-03).

Downloads WHAM's checkpoints + auxiliary body-model files and uploads them to
the Modal volume `wham-checkpoints`, reusing the ViTPose/YOLO/SMPL files
already present in local_checkpoints/ from the GVHMR setup.

Run: python -m analyzer_3d.src.wham_setup

Volume layout produced (matches what wham_modal.py symlinks into /wham):
    checkpoints/wham_vit_bedlam_w_3dpw.pth.tar
    checkpoints/hmr2a.ckpt
    checkpoints/vitpose-h-multi-coco.pth
    checkpoints/yolov8x.pt
    body_models/coco_aug_dict.pth, smpl_faces.npy, smpl_mean_params.npz,
                J_regressor_*.npy, smplx2smpl.pkl, ...
    body_models/smpl/SMPL_NEUTRAL.pkl
"""

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

# Google Drive IDs from WHAM's fetch_demo_data.sh
GDRIVE_FILES = {
    "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar": "19qkI-a6xuwob9_RFNSPWf1yWErwVVlks",
    "checkpoints/hmr2a.ckpt": "1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y",
}
BODY_MODELS_TARBALL_ID = "1pbmzRbWGgae6noDIyQOnohzaVnX_csUZ"

LOCAL_DIR = Path("local_checkpoints_wham")
GVHMR_LOCAL = Path("local_checkpoints")


def gdown_file(file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  gdown {file_id} -> {dest}")
    subprocess.run(
        [sys.executable, "-m", "gdown", "--id", file_id, "-O", str(dest)],
        check=True,
    )


def download():
    print("\n=== Downloading WHAM checkpoints (Google Drive) ===\n")
    for rel, fid in GDRIVE_FILES.items():
        gdown_file(fid, LOCAL_DIR / rel)

    tarball = LOCAL_DIR / "body_models.tar.gz"
    gdown_file(BODY_MODELS_TARBALL_ID, tarball)

    bm_dir = LOCAL_DIR / "body_models"
    if not (bm_dir / "smpl_mean_params.npz").exists():
        print("  Extracting body_models.tar.gz...")
        with tarfile.open(tarball) as tf:
            tf.extractall(LOCAL_DIR)
        # tarball may extract as dataset/body_models or body_models
        nested = LOCAL_DIR / "dataset" / "body_models"
        if nested.exists() and not (bm_dir / "smpl_mean_params.npz").exists():
            bm_dir.mkdir(parents=True, exist_ok=True)
            for item in nested.iterdir():
                shutil.move(str(item), str(bm_dir / item.name))
    print("  body_models contents:", sorted(p.name for p in bm_dir.iterdir()) if bm_dir.exists() else "MISSING")


def reuse_gvhmr_files():
    print("\n=== Reusing GVHMR-local checkpoint files ===\n")
    pairs = [
        (GVHMR_LOCAL / "vitpose" / "vitpose-h-multi-coco.pth", LOCAL_DIR / "checkpoints" / "vitpose-h-multi-coco.pth"),
        (GVHMR_LOCAL / "yolo" / "yolov8x.pt", LOCAL_DIR / "checkpoints" / "yolov8x.pt"),
        (GVHMR_LOCAL / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl", LOCAL_DIR / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl"),
    ]
    for src, dest in pairs:
        if dest.exists():
            print(f"  Already staged: {dest}")
            continue
        if not src.exists():
            raise FileNotFoundError(f"Expected GVHMR local file missing: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"  Copied {src} -> {dest}")


def upload():
    import modal
    from pathlib import PurePosixPath

    print("\n=== Uploading to Modal volume 'wham-checkpoints' ===\n")
    vol = modal.Volume.from_name("wham-checkpoints", create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for fp in sorted(LOCAL_DIR.rglob("*")):
            if fp.is_file() and fp.name != "body_models.tar.gz":
                rel = fp.relative_to(LOCAL_DIR)
                if rel.parts[0] == "dataset":  # leftover extraction dir
                    continue
                remote = "/" + PurePosixPath(rel).as_posix()
                print(f"  {rel} ({fp.stat().st_size / 1e6:.1f} MB) -> {remote}")
                batch.put_file(str(fp), remote)
    print("\n  Upload complete.")


def main():
    download()
    reuse_gvhmr_files()
    upload()
    print("\nNext: python -m modal deploy analyzer_3d/src/wham_modal.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
