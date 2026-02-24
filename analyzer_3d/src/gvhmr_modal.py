"""GVHMR pose estimation running on Modal cloud GPU.

Deploys GVHMR (Gravity-aware Video Human Motion Recovery) to a Modal T4 GPU
for extracting 3D human pose (SMPL parameters) from breakdancing videos.

Prerequisites:
    1. Register at smpl.is.tue.mpg.de and smpl-x.is.tue.mpg.de
    2. Download SMPL_NEUTRAL.pkl and SMPLX_NEUTRAL.npz
    3. Run the setup script to upload checkpoints to Modal:
       python -m analyzer_3d.src.gvhmr_setup

Usage (deploy):
    python -m modal deploy analyzer_3d/src/gvhmr_modal.py

Usage (client):
    from analyzer_3d.src.gvhmr_modal import estimate_pose_gvhmr
    result = estimate_pose_gvhmr("video.mp4")
"""

import modal

app = modal.App("gvhmr-pose-estimation")

# Persistent volume for model checkpoints (too large for image)
checkpoints_volume = modal.Volume.from_name(
    "gvhmr-checkpoints", create_if_missing=True
)

# Container image with all GVHMR dependencies
gvhmr_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0",
        "wget", "unzip", "python3-tk",
    )
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl",
    )
    .pip_install(
        "timm==0.9.12",
        "lightning==2.3.0",
        "hydra-core==1.3",
        "hydra-zen",
        "hydra_colorlog",
        "rich",
        "numpy==1.23.5",
        "matplotlib",
        "setuptools>=68.0",
        "tensorboardX",
        "opencv-python",
        "ffmpeg-python",
        "scikit-image",
        "termcolor",
        "einops",
        "imageio==2.34.1",
        "av==13.0.0",
        "joblib",
        "trimesh",
        "chumpy",
        "smplx",
        "ultralytics==8.2.42",
        "cython_bbox",
        "lapx",
    )
    .run_commands(
        "git clone https://github.com/zju3dv/GVHMR /gvhmr",
        "cd /gvhmr && pip install -e .",
    )
    # Stub out modules only needed for training/visualization (not inference)
    .run_commands(
        "mkdir -p /usr/local/lib/python3.10/site-packages/wis3d",
        "mkdir -p /usr/local/lib/python3.10/site-packages/pycolmap",
        "echo 'class Wis3D: pass' > /usr/local/lib/python3.10/site-packages/wis3d/__init__.py",
        "echo '' > /usr/local/lib/python3.10/site-packages/pycolmap/__init__.py",
    )
    .env({"HYDRA_FULL_ERROR": "1"})
)


@app.cls(
    image=gvhmr_image,
    gpu="T4",
    timeout=900,
    scaledown_window=300,
    min_containers=0,
    volumes={"/checkpoints": checkpoints_volume},
)
class GVHMREstimator:
    """GVHMR model for 3D human pose estimation from video."""

    @modal.enter()
    def load_models(self):
        import os
        import sys

        import torch

        os.chdir("/gvhmr")
        sys.path.insert(0, "/gvhmr")

        # Symlink checkpoints from volume into expected location
        ckpt_src = "/checkpoints"
        ckpt_dst = "/gvhmr/inputs/checkpoints"
        if not os.path.exists(ckpt_dst):
            os.makedirs("/gvhmr/inputs", exist_ok=True)
            os.symlink(ckpt_src, ckpt_dst)

        # Verify required files exist
        required_files = [
            "gvhmr/gvhmr_siga24_release.ckpt",
            "hmr2/epoch=10-step=25000.ckpt",
            "vitpose/vitpose-h-multi-coco.pth",
            "yolo/yolov8x.pt",
            "body_models/smplx/SMPLX_NEUTRAL.npz",
            "body_models/smpl/SMPL_NEUTRAL.pkl",
        ]
        missing = [f for f in required_files if not os.path.exists(f"{ckpt_src}/{f}")]
        if missing:
            raise RuntimeError(
                f"Missing checkpoint files in volume 'gvhmr-checkpoints':\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\n\nRun: python -m analyzer_3d.src.gvhmr_setup"
            )

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Load preprocessing components
        from hmr4d.utils.preproc import Tracker, VitPoseExtractor, Extractor

        print("Loading YOLO tracker...")
        self.tracker = Tracker()

        print("Loading ViTPose extractor...")
        self.vitpose = VitPoseExtractor()

        print("Loading HMR2 feature extractor...")
        self.feature_extractor = Extractor()

        # Load GVHMR model — bypass Hydra config system entirely
        # (store_gvhmr.py imports training-only deps that are hard to satisfy)
        print("Loading GVHMR model...")
        from omegaconf import OmegaConf
        from hydra.utils import instantiate

        # Build config matching what Hydra would produce from demo.yaml +
        # structured configs registered by store_gvhmr.py
        pipeline_cfg = OmegaConf.create({
            "_target_": "hmr4d.model.gvhmr.pipeline.gvhmr_pipeline.Pipeline",
            "args_denoiser3d": {
                "_target_": "hmr4d.network.gvhmr.relative_transformer.NetworkEncoderRoPE",
                # All defaults from NetworkEncoderRoPE.__init__
                "output_dim": 151,
                "max_len": 120,
                "cliffcam_dim": 3,
                "cam_angvel_dim": 6,
                "imgseq_dim": 1024,
                "latent_dim": 512,
                "num_layers": 12,
                "num_heads": 8,
                "mlp_ratio": 4.0,
                "pred_cam_dim": 3,
                "static_conf_dim": 6,
                "dropout": 0.1,
                "avgbeta": True,
            },
            "args": {
                "endecoder_opt": {
                    "_target_": "hmr4d.model.gvhmr.utils.endecoder.EnDecoder",
                    "stats_name": "MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                },
                "normalize_cam_angvel": True,
                "weights": None,
                "static_conf": None,
            },
        })

        model_cfg = OmegaConf.create({
            "_target_": "hmr4d.model.gvhmr.gvhmr_pl_demo.DemoPL",
            "pipeline": pipeline_cfg,
        })

        ckpt_path = "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"

        self.model = instantiate(model_cfg, _recursive_=False)
        self.model.load_pretrained_model(ckpt_path)
        self.model = self.model.eval().cuda()

        self.device = torch.device("cuda")
        print("All models loaded successfully!")

    @modal.method()
    def estimate_pose(
        self,
        video_bytes: bytes,
        static_cam: bool = True,
    ) -> bytes:
        """Estimate 3D human pose from a video.

        Args:
            video_bytes: Raw video file bytes (MP4)
            static_cam: True for static/tripod camera (recommended for tutorials)

        Returns:
            Pickled dict with SMPL parameters:
            {
                "smpl_params_global": {
                    "body_pose": (F, 63),     # 21 joints x 3 axis-angle
                    "global_orient": (F, 3),  # root rotation
                    "transl": (F, 3),         # root translation
                    "betas": (F, 10),         # shape parameters
                },
                "smpl_params_incam": { ... },  # same keys, camera coords
                "fps": float,
                "num_frames": int,
            }
        """
        import os
        import pickle
        import tempfile

        import cv2
        import numpy as np
        import torch

        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
        from hmr4d.utils.geo_transform import compute_cam_angvel

        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Stage 1: Person detection & tracking (YOLO)
        print("Stage 1/4: Person detection (YOLO)...")
        bbx_xyxy = self.tracker.get_one_track(video_path).float()
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

        # Stage 2: 2D Pose estimation (ViTPose)
        print("Stage 2/4: 2D pose estimation (ViTPose)...")
        kp2d = self.vitpose.extract(video_path, bbx_xys)

        # Stage 3: Feature extraction (HMR2)
        print("Stage 3/4: Feature extraction (HMR2)...")
        f_imgseq = self.feature_extractor.extract_video_features(video_path, bbx_xys)

        # Build data dict for GVHMR (following demo.py's load_data_dict)
        length = total_frames

        # Camera intrinsics (assume default focal length)
        focal_length = max(width, height)
        K_fullimg = torch.zeros(length, 3, 3)
        K_fullimg[:, 0, 0] = focal_length
        K_fullimg[:, 1, 1] = focal_length
        K_fullimg[:, 0, 2] = width / 2
        K_fullimg[:, 1, 2] = height / 2
        K_fullimg[:, 2, 2] = 1.0

        # Camera angular velocity
        if static_cam:
            # Static camera: identity rotation for all frames
            R_w2c = torch.eye(3).unsqueeze(0).repeat(length, 1, 1)
        else:
            # For moving camera, would need DPVO/SimpleVO
            # Fall back to identity for now
            R_w2c = torch.eye(3).unsqueeze(0).repeat(length, 1, 1)

        cam_angvel = compute_cam_angvel(R_w2c)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": bbx_xys,
            "kp2d": kp2d,
            "K_fullimg": K_fullimg,
            "cam_angvel": cam_angvel,
            "f_imgseq": f_imgseq,
        }

        # Stage 4: GVHMR inference
        # (model.predict handles batching and CUDA transfer internally)
        print("Stage 4/4: GVHMR 3D pose estimation...")
        pred = self.model.predict(data, static_cam=static_cam)

        # Detach and move to CPU
        def detach_to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            elif isinstance(x, dict):
                return {k: detach_to_cpu(v) for k, v in x.items()}
            return x

        result = {
            "smpl_params_global": detach_to_cpu(pred["smpl_params_global"]),
            "smpl_params_incam": detach_to_cpu(pred["smpl_params_incam"]),
            "fps": fps,
            "num_frames": total_frames,
        }

        # Extract SMPL body mesh for GLB export (6890 vertices, proper skinning)
        try:
            import smplx

            smpl_model = smplx.create(
                "inputs/checkpoints/body_models",
                model_type="smpl",
                gender="neutral",
                batch_size=1,
            )

            # Get average betas across all frames for consistent body shape
            betas_avg = torch.tensor(
                result["smpl_params_global"]["betas"].mean(axis=0),
                dtype=torch.float32,
            ).unsqueeze(0)

            # Forward pass with zero pose to get shaped rest mesh
            smpl_output = smpl_model(
                betas=betas_avg,
                body_pose=torch.zeros(1, 69),   # rest pose
                global_orient=torch.zeros(1, 3),
            )

            # Get rest pose joint positions (critical for correct bone placement)
            joints = smpl_output.joints[0, :24].detach().cpu().numpy()  # (24, 3)

            result["mesh"] = {
                "vertices": smpl_output.vertices[0].detach().cpu().numpy(),  # (6890, 3)
                "faces": smpl_model.faces.astype(np.int32),                 # (13776, 3)
                "weights": smpl_model.lbs_weights.detach().cpu().numpy(),   # (6890, 24)
                "joints": joints,                                           # (24, 3)
            }
            print(f"SMPL mesh: {result['mesh']['vertices'].shape[0]} vertices, "
                  f"{result['mesh']['faces'].shape[0]} faces, "
                  f"{joints.shape[0]} joints")
        except Exception as e:
            print(f"Warning: Could not extract SMPL mesh: {e}")
            print("GLB will use stick figure fallback")

        # Cleanup
        os.unlink(video_path)

        # Serialize
        output = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Done! Output: {len(output) / 1024:.1f} KB")
        return output


def estimate_pose_gvhmr(
    input_path: str,
    output_path: str | None = None,
    static_cam: bool = True,
) -> str:
    """Client-side function to estimate 3D pose using GVHMR on Modal.

    Reads a local video, sends it to Modal cloud GPU, and saves the result.

    Args:
        input_path: Path to local input video
        output_path: Path for output pickle file (auto-generated if None)
        static_cam: True for static/tripod camera

    Returns:
        Path to the output pickle file containing SMPL parameters
    """
    from pathlib import Path

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".pkl")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"GVHMR Pose Estimation (Modal Cloud GPU)")
    print(f"  Input: {input_path}")
    print(f"  Static camera: {static_cam}")
    print(f"  Uploading {input_path.stat().st_size / 1024 / 1024:.1f} MB...")

    # Read local file
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    # Call Modal function
    estimator = modal.Cls.from_name("gvhmr-pose-estimation", "GVHMREstimator")()
    output_bytes = estimator.estimate_pose.remote(
        input_bytes, static_cam=static_cam
    )

    # Write result
    with open(output_path, "wb") as f:
        f.write(output_bytes)

    print(f"  Saved SMPL parameters: {output_path}")
    print(f"  Output size: {output_path.stat().st_size / 1024:.1f} KB")

    return str(output_path)
