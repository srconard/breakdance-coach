"""WHAM pose estimation on Modal cloud GPU (mocap model spike, 2026-07-03).

WHAM (CVPR 2024) is contact-aware: it predicts foot-ground contact and uses it
to reconstruct the world-frame trajectory. This spike tests it against GVHMR
on breaking footage, where GVHMR leaves support hands floating.

Prerequisites:
    python -m analyzer_3d.src.wham_setup          # upload checkpoints once
    python -m modal deploy analyzer_3d/src/wham_modal.py

Usage (client):
    from analyzer_3d.src.wham_modal import estimate_pose_wham
    estimate_pose_wham("video.mp4", "video_wham_smpl.pkl")

Output pickle uses the same schema as gvhmr_modal.py, so motion_cleanup,
the Blender exporter, and all diagnostics work unchanged.
"""

import modal

app = modal.App("wham-pose-estimation")

checkpoints_volume = modal.Volume.from_name("wham-checkpoints", create_if_missing=True)

# Match WHAM's tested stack (docs/INSTALL.md): torch 1.11/cu113 (their conda env
# uses py3.9, but Modal's builder starts at 3.10 and torch 1.11 ships cp310 wheels).
# mmcv 1.3.9 imports torch._six, which newer torch removed — do not bump torch.
# mmpose is provided by WHAM's third-party/ViTPose submodule (editable install).
wham_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget", "unzip")
    .pip_install(
        "torch==1.11.0+cu113",
        "torchvision==0.12.0+cu113",
        index_url="https://download.pytorch.org/whl/cu113",
    )
    .pip_install("numpy==1.22.3", "setuptools==59.5.0")
    .pip_install(
        "chumpy @ git+https://github.com/mattloper/chumpy",
        "yacs",
        "joblib",
        "scikit-image",
        "opencv-python",
        "imageio[ffmpeg]",
        "matplotlib",
        "tensorboard",
        "smplx",
        "progress",
        "einops",
        "mmcv==1.3.9",
        "timm==0.4.9",
        "munkres",
        "xtcocotools>=1.8",
        "loguru",
        "tqdm",
        "ultralytics==8.2.42",
    )
    .run_commands(
        "git clone --recursive https://github.com/yohanshin/WHAM /wham",
        "pip install -v -e /wham/third-party/ViTPose",
    )
)


@app.cls(
    image=wham_image,
    gpu="T4",
    timeout=1800,
    scaledown_window=300,
    min_containers=0,
    volumes={"/checkpoints": checkpoints_volume},
)
class WHAMEstimator:
    @modal.enter()
    def setup(self):
        import os

        os.chdir("/wham")
        # WHAM resolves everything relative to the repo root
        if not os.path.exists("/wham/checkpoints"):
            os.symlink("/checkpoints/checkpoints", "/wham/checkpoints")
        os.makedirs("/wham/dataset", exist_ok=True)
        if not os.path.exists("/wham/dataset/body_models"):
            os.symlink("/checkpoints/body_models", "/wham/dataset/body_models")

        required = [
            "/wham/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
            "/wham/checkpoints/hmr2a.ckpt",
            "/wham/checkpoints/vitpose-h-multi-coco.pth",
            "/wham/checkpoints/yolov8x.pt",
            "/wham/dataset/body_models/smpl_mean_params.npz",
            "/wham/dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
        ]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            raise RuntimeError(
                "Missing files in volume 'wham-checkpoints':\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\nRun: python -m analyzer_3d.src.wham_setup"
            )

    @modal.method()
    def estimate_pose(self, video_bytes: bytes, static_cam: bool = True) -> bytes:
        """Run WHAM demo.py on a video; return our standard SMPL pickle bytes.

        static_cam is accepted for interface parity; without DPVO installed,
        WHAM's demo already assumes an identity (static) camera.
        """
        import os
        import pickle
        import subprocess
        import tempfile

        import cv2
        import joblib
        import numpy as np
        import torch

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as f:
            f.write(video_bytes)
            video_path = f.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Video: {total_frames} frames @ {fps:.2f} fps")

        out_dir = tempfile.mkdtemp(prefix="wham_out_", dir="/tmp")
        cmd = [
            "python", "demo.py",
            "--video", video_path,
            "--output_pth", out_dir,
            "--save_pkl",
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd="/wham", capture_output=True, text=True)
        print(proc.stdout[-3000:])
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
            raise RuntimeError(f"WHAM demo failed (exit {proc.returncode})")

        # demo.py writes <output_pth>/<video_stem>/wham_output.pkl
        stem = os.path.splitext(os.path.basename(video_path))[0]
        pkl_path = os.path.join(out_dir, stem, "wham_output.pkl")
        if not os.path.exists(pkl_path):
            # fall back to a recursive search
            hits = []
            for root, _, files in os.walk(out_dir):
                hits += [os.path.join(root, fn) for fn in files if fn == "wham_output.pkl"]
            if not hits:
                raise RuntimeError(f"wham_output.pkl not found under {out_dir}")
            pkl_path = hits[0]

        results = joblib.load(pkl_path)
        # one entry per tracked subject — take the one with the most frames
        sid = max(results.keys(), key=lambda k: len(results[k]["frame_ids"]))
        sub = results[sid]
        frame_ids = np.asarray(sub["frame_ids"])
        pose_world = np.asarray(sub["pose_world"])   # (F, 72) axis-angle, 24 joints
        trans_world = np.asarray(sub["trans_world"])  # (F, 3)
        betas = np.asarray(sub["betas"])              # (F, 10) or (10,)
        if betas.ndim == 1:
            betas = np.tile(betas[None], (pose_world.shape[0], 1))
        print(f"Subject {sid}: {pose_world.shape[0]} frames "
              f"(video frames {frame_ids.min()}..{frame_ids.max()})")

        result = {
            "smpl_params_global": {
                "body_pose": pose_world[:, 3:66].astype(np.float32),   # joints 1-21
                "global_orient": pose_world[:, :3].astype(np.float32),
                "transl": trans_world.astype(np.float32),
                "betas": betas.astype(np.float32),
            },
            "fps": float(fps),
            "num_frames": int(pose_world.shape[0]),
            "frame_ids": frame_ids,
            "backend": "wham",
        }

        # Rest mesh block (same as gvhmr_modal) so cleanup + Blender export work
        try:
            import smplx

            smpl_model = smplx.create(
                "/wham/dataset/body_models", model_type="smpl",
                gender="neutral", batch_size=1,
            )
            betas_avg = torch.tensor(betas.mean(axis=0), dtype=torch.float32).unsqueeze(0)
            out = smpl_model(
                betas=betas_avg,
                body_pose=torch.zeros(1, 69),
                global_orient=torch.zeros(1, 3),
            )
            result["mesh"] = {
                "vertices": out.vertices[0].detach().cpu().numpy(),
                "faces": smpl_model.faces.astype(np.int32),
                "weights": smpl_model.lbs_weights.detach().cpu().numpy(),
                "joints": out.joints[0, :24].detach().cpu().numpy(),
            }
            print("SMPL rest mesh attached")
        except Exception as e:
            print(f"Warning: no mesh block ({e}) — stick figure fallback will be used")

        os.unlink(video_path)
        payload = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done: {len(payload) / 1024:.0f} KB")
        return payload


def estimate_pose_wham(input_path: str, output_path: str | None = None, static_cam: bool = True) -> str:
    """Client: send a local video to WHAM on Modal, save the SMPL pickle."""
    from pathlib import Path

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".wham.pkl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"WHAM Pose Estimation (Modal T4)")
    print(f"  Uploading {input_path.stat().st_size / 1e6:.1f} MB...")
    estimator = modal.Cls.from_name("wham-pose-estimation", "WHAMEstimator")()
    out = estimator.estimate_pose.remote(input_path.read_bytes(), static_cam=static_cam)
    output_path.write_bytes(out)
    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return str(output_path)
