"""RIFE frame interpolation running on Modal cloud GPU.

Deploys Practical-RIFE v4.25 to a Modal T4 GPU for high-quality
frame interpolation of breakdance tutorial clips.

Usage (deploy first time):
    python -m modal deploy src/rife_modal.py

Usage (called from interpolate.py, or directly):
    python -c "from src.rife_modal import interpolate_video_rife; \
        interpolate_video_rife('clip.mp4', multi=3)"
"""

import modal

app = modal.App("rife-interpolation")

# Container image with all RIFE dependencies
rife_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "git")
    .apt_install("unzip", "wget")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy==1.23.5",
        "opencv-python==4.10.0.84",
        "sk-video==1.1.10",
        "tqdm",
    )
    .run_commands(
        "git clone https://github.com/hzwer/Practical-RIFE.git /root/rife",
        "cd /root/rife && mkdir -p train_log",
        # Download RIFE v4.25 model weights from Hugging Face mirror
        'wget -O /tmp/rife_v4.25.zip "https://huggingface.co/r3gm/RIFE/resolve/main/RIFEv4.25_0919.zip"',
        "unzip /tmp/rife_v4.25.zip -d /tmp/rife_model/",
        "cp /tmp/rife_model/train_log/*.py /tmp/rife_model/train_log/*.pkl /root/rife/train_log/",
        "rm -rf /tmp/rife_v4.25.zip /tmp/rife_model",
        "ls -la /root/rife/train_log/",
    )
)


@app.cls(
    image=rife_image,
    gpu="T4",
    timeout=600,
    scaledown_window=300,
    min_containers=0,
)
class RIFEInterpolator:
    """RIFE model loaded once per container, reused across requests."""

    @modal.enter()
    def load_model(self):
        import os
        import sys

        import torch

        os.chdir("/root/rife")
        sys.path.insert(0, "/root/rife")

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from train_log.RIFE_HDv3 import Model

        self.model = Model()
        if not hasattr(self.model, "version"):
            self.model.version = 0
        self.model.load_model("train_log", -1)
        self.model.eval()
        self.model.device()
        self.device = torch.device("cuda")
        print(f"RIFE model loaded (version {self.model.version})")

    @modal.method()
    def interpolate(
        self,
        input_bytes: bytes,
        multi: int = 2,
        scale: float = 1.0,
    ) -> bytes:
        """Interpolate a video, returning the result as bytes.

        Args:
            input_bytes: Raw MP4 file bytes
            multi: Frame rate multiplier (2 = 2x frames, 3 = 3x, etc.)
            scale: Resolution scale (1.0 for HD, 0.5 for 4K)

        Returns:
            Interpolated MP4 file as bytes
        """
        import subprocess
        import tempfile

        import cv2
        import numpy as np
        import torch
        from torch.nn import functional as F

        # Write input to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(input_bytes)
            input_path = f.name

        raw_output = input_path.replace(".mp4", "_raw.mp4")
        final_output = input_path.replace(".mp4", "_rife.mp4")

        # Read video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Write at original fps — with multi*N frames, this creates slow-motion
        # (e.g. 3x frames at 12fps = video plays 3x slower)
        out_fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_output, fourcc, out_fps, (w, h))

        # Padding for RIFE (dimensions must be divisible by 128/scale)
        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        def to_tensor(frame):
            t = (
                torch.from_numpy(np.transpose(frame[:, :, ::-1].copy(), (2, 0, 1)))
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
                .float()
                / 255.0
            )
            return F.pad(t, padding)

        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read input video")

        frame_idx = 0
        print(f"Processing {total_frames} frames at {multi}x interpolation...")

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            # Write original frame
            out.write(prev_frame)

            # Generate intermediate frames
            I0 = to_tensor(prev_frame)
            I1 = to_tensor(curr_frame)

            for i in range(multi - 1):
                timestep = (i + 1) / multi
                if self.model.version >= 3.9:
                    mid = self.model.inference(I0, I1, timestep, scale)
                else:
                    mid = self.model.inference(I0, I1, scale)

                mid_np = (
                    (mid[0] * 255.0)
                    .byte()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)[:h, :w]
                )
                out.write(mid_np[:, :, ::-1].copy())

            prev_frame = curr_frame
            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"  Frame {frame_idx}/{total_frames}")

        # Write last frame
        out.write(prev_frame)
        cap.release()
        out.release()

        print(f"Interpolation done. Re-encoding with H.264 at {fps}fps (slow-mo)...")

        # Re-encode with H.264, forcing original fps for slow-motion effect
        # (3x interpolated frames at original fps = 3x longer video)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_output,
                "-r",
                str(int(fps)),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                "-an",
                final_output,
            ],
            check=True,
            capture_output=True,
        )

        with open(final_output, "rb") as f:
            result = f.read()

        # Cleanup
        import os

        for p in [input_path, raw_output, final_output]:
            if os.path.exists(p):
                os.unlink(p)

        print(f"Done! Output: {len(result) / 1024 / 1024:.1f} MB")
        return result


def interpolate_video_rife(
    input_path: str,
    output_path: str | None = None,
    multi: int = 2,
    scale: float = 1.0,
) -> str:
    """Client-side function to interpolate a video using RIFE on Modal.

    Reads a local video, sends it to Modal cloud GPU, and saves the result.

    Args:
        input_path: Path to local input video
        output_path: Path for output (auto-generated if None)
        multi: Frame rate multiplier (2, 3, 4, etc.)
        scale: Resolution scale (1.0 for HD, 0.5 for 4K)

    Returns:
        Path to the output video
    """
    from pathlib import Path

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_rife_{multi}x{input_path.suffix}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"RIFE Interpolation (Modal Cloud GPU)")
    print(f"  Input: {input_path}")
    print(f"  Multiplier: {multi}x")
    print(f"  Uploading {input_path.stat().st_size / 1024:.0f} KB...")

    # Read local file
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    # Call Modal function
    interpolator = modal.Cls.from_name("rife-interpolation", "RIFEInterpolator")()
    output_bytes = interpolator.interpolate.remote(
        input_bytes, multi=multi, scale=scale
    )

    # Write result
    with open(output_path, "wb") as f:
        f.write(output_bytes)

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)

    print(f"  Input:  {input_size:.1f} MB")
    print(f"  Output: {output_size:.1f} MB ({multi}x frames)")
    print(f"  Saved: {output_path}")

    return str(output_path)
