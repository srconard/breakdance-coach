"""DECO dense human-scene contact estimation on Modal cloud GPU.

Deploys DECO (Dense Estimation of 3D Human-Scene Contact In The Wild,
ICCV 2023 oral, github.com/sha2nkt/deco) to a Modal L4 GPU. For each
sampled video frame it returns the model's native output: a per-vertex
contact probability over the SMPL 6890-vertex mesh (which body vertices
are touching the scene). This is the machine behind the KNOW-layer
"contact signature" for the breakdance-coach move library.

Why this design (differs from GVHMR/WHAM):
    * DECO's checkpoints are PUBLIC (keeper.mpdl.mpg.de, no registration),
      so we download them directly on Modal into a volume — no local
      SMPL registration / upload step is needed.
    * The contact-vector forward pass (`cont, _, _ = model(img)`) needs
      ONLY deco_best.pth. SMPL body-model .pkl files are used solely by
      DECO's optional visualization mesh, which we do not run. So none of
      the registration-gated SMPL assets are on the critical path.
    * DECO's stock inference.py resizes the *full frame* to 256x256. On
      AIST++ the dancer fills ~15% of a 1080p frame -> a ~40px person,
      which is out-of-distribution for a model trained on person-centric
      imagery. We add a YOLO person-crop (square, padded) so DECO gets a
      fair shot; the bbox used is returned per frame for auditing.

Setup (one-time, downloads ~1.4GB checkpoint into a Modal volume):
    python -m modal run analyzer_3d.src.deco_modal::setup_checkpoints

Deploy:
    python -m modal deploy analyzer_3d/src/deco_modal.py

Client:
    from analyzer_3d.src.deco_modal import estimate_contacts_deco
    estimate_contacts_deco("video.mp4", "out.npz", sample_fps=5.0)
"""

import modal

app = modal.App("deco-contact-estimation")

# Public DECO checkpoints live here (no registration gate; verified 2026-07-09).
checkpoints_volume = modal.Volume.from_name("deco-checkpoints", create_if_missing=True)

RELEASE_CHECKPOINT_URL = "https://keeper.mpdl.mpg.de/f/6f2e2258558f46ceb269/?dl=1"

# Container image. DECO's contact forward pass only touches models/deco.py ->
# components.py -> utils/hrnet.py (pure torch/torchvision). It does NOT import
# pyrender/trimesh/pytorch3d/monai/smplx (those are viz/training only), so we
# skip them and reuse GVHMR's proven torch 2.3 cu121 stack.
deco_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy==1.23.5",
        "opencv-python-headless",
        "loguru",
        "yacs",
        "scikit-image",
        "einops",
        "ultralytics==8.2.42",
    )
    .run_commands(
        "git clone https://github.com/sha2nkt/deco /deco",
        # Pre-fetch the small YOLO person detector (public ultralytics asset)
        # so runtime containers don't each download it.
        "cd /deco && python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"",
    )
    .env({"YOLO_CONFIG_DIR": "/tmp/ultralytics"})
)


@app.function(
    image=deco_image,
    volumes={"/checkpoints": checkpoints_volume},
    timeout=1800,
)
def setup_checkpoints():
    """Download DECO's public release checkpoint into the volume (one-time).

    Extracts only deco_best.pth — the full contact model, loaded strict=True.
    """
    import os
    import subprocess
    import glob

    dst = "/checkpoints/deco_best.pth"
    if os.path.exists(dst):
        print(f"Checkpoint already present: {dst}")
        return

    os.makedirs("/checkpoints", exist_ok=True)
    tmp_tar = "/tmp/Release_Checkpoint.tar.gz"
    print("Downloading Release_Checkpoint.tar.gz (~1.4GB) ...")
    subprocess.run(
        ["wget", "-q", "--max-redirect=2", "--trust-server-names",
         RELEASE_CHECKPOINT_URL, "-O", tmp_tar],
        check=True,
    )
    print("Extracting ...")
    os.makedirs("/tmp/ckpt_extract", exist_ok=True)
    subprocess.run(["tar", "-xf", tmp_tar, "-C", "/tmp/ckpt_extract"], check=True)

    # Find deco_best.pth anywhere in the extracted tree.
    hits = glob.glob("/tmp/ckpt_extract/**/deco_best.pth", recursive=True)
    if not hits:
        all_pth = glob.glob("/tmp/ckpt_extract/**/*.pth", recursive=True)
        raise RuntimeError(
            "deco_best.pth not found in Release_Checkpoint.tar.gz. "
            f"Found .pth files: {all_pth}"
        )
    import shutil
    shutil.copyfile(hits[0], dst)
    checkpoints_volume.commit()
    size_mb = os.path.getsize(dst) / 1024 / 1024
    print(f"Saved {dst} ({size_mb:.1f} MB) from {hits[0]}")


@app.cls(
    image=deco_image,
    gpu="L4",
    timeout=1200,
    scaledown_window=240,
    min_containers=0,
    volumes={"/checkpoints": checkpoints_volume},
)
class DECOEstimator:
    """DECO per-frame vertex-contact estimation from video."""

    @modal.enter()
    def load_models(self):
        import os
        import sys

        import torch

        os.chdir("/deco")
        sys.path.insert(0, "/deco")

        ckpt = "/checkpoints/deco_best.pth"
        if not os.path.exists(ckpt):
            raise RuntimeError(
                "Missing /checkpoints/deco_best.pth. Run: "
                "python -m modal run analyzer_3d.src.deco_modal::setup_checkpoints"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        from models.deco import DECO

        # ('hrnet', context=True) matches the released deco_best.pth. HRNet's
        # own pretrained file is absent, but deco_best.pth is loaded strict=True
        # over the whole model, so every weight/buffer is set from it.
        self.model = DECO("hrnet", True, self.device)
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["deco"], strict=True)
        self.model.eval()
        torch.set_grad_enabled(False)

        # YOLO person detector for fair person-centric crops.
        from ultralytics import YOLO
        self.yolo = YOLO("yolov8n.pt")

        print("DECO + YOLO loaded.")

    def _person_bbox(self, frame_bgr):
        """Return the highest-confidence person box (xyxy) or None."""
        res = self.yolo.predict(
            frame_bgr, classes=[0], verbose=False, device=self.device
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None
        import numpy as np
        confs = res.boxes.conf.detach().cpu().numpy()
        xyxy = res.boxes.xyxy.detach().cpu().numpy()
        return xyxy[int(np.argmax(confs))]

    @staticmethod
    def _square_crop(frame_bgr, xyxy, pad_frac=0.30):
        """Square, padded crop around xyxy; clamps to frame bounds."""
        import numpy as np

        h, w = frame_bgr.shape[:2]
        if xyxy is None:
            # Center square fallback.
            s = min(h, w)
            cx, cy = w / 2, h / 2
            x0 = int(cx - s / 2); y0 = int(cy - s / 2)
            return frame_bgr[y0:y0 + s, x0:x0 + s], np.array([x0, y0, x0 + s, y0 + s])

        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(bw, bh) * (1.0 + 2 * pad_frac)
        x0 = int(round(cx - side / 2)); y0 = int(round(cy - side / 2))
        x3 = int(round(cx + side / 2)); y3 = int(round(cy + side / 2))
        x0c, y0c = max(0, x0), max(0, y0)
        x3c, y3c = min(w, x3), min(h, y3)
        crop = frame_bgr[y0c:y3c, x0c:x3c]
        return crop, np.array([x0c, y0c, x3c, y3c])

    @modal.method()
    def estimate_contacts(
        self,
        video_bytes: bytes,
        sample_fps: float = 5.0,
        crop: bool = True,
        batch_size: int = 16,
    ) -> bytes:
        """Run DECO on sampled frames.

        Returns npz bytes with:
            contact:       (T, 6890) float16  per-vertex contact probability
            times:         (T,)      float32  timestamp (s) of each sampled frame
            frame_idx:     (T,)      int32    source frame index
            bbox:          (T, 4)    int32    crop xyxy used (-1 if none/no-crop)
            had_person:    (T,)      bool     YOLO found a person
            sample_fps, video_fps, num_source_frames, width, height, crop
        """
        import io
        import os
        import tempfile

        import cv2
        import numpy as np
        import torch

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {width}x{height} @ {video_fps:.2f}fps, {n_src} frames")

        step = max(1, int(round(video_fps / float(sample_fps))))
        want_idx = list(range(0, n_src, step))
        want_set = set(want_idx)

        crops, times, fidx, bboxes, had_person = [], [], [], [], []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in want_set:
                if crop:
                    bb = self._person_bbox(frame)
                    c, used = self._square_crop(frame, bb)
                    had_person.append(bb is not None)
                    bboxes.append(used.astype(np.int32))
                else:
                    c = frame
                    had_person.append(False)
                    bboxes.append(np.array([-1, -1, -1, -1], dtype=np.int32))
                if c.size == 0:
                    c = frame
                crops.append(c)
                times.append(i / video_fps)
                fidx.append(i)
            i += 1
        cap.release()
        os.unlink(video_path)

        T = len(crops)
        print(f"Sampled {T} frames at ~{sample_fps}fps (step={step}); crop={crop}")

        # Preprocess exactly as DECO inference.py: resize 256, /255, CHW, no
        # mean/std normalization. (Model was trained with this convention.)
        def prep(img_bgr):
            img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1) / 255.0
            return img.astype(np.float32)

        contacts = np.zeros((T, 6890), dtype=np.float16)
        for b in range(0, T, batch_size):
            batch = np.stack([prep(c) for c in crops[b:b + batch_size]], axis=0)
            t = torch.from_numpy(batch).to(self.device)
            cont, _, _ = self.model(t)          # (B, 6890) sigmoid probs
            contacts[b:b + batch_size] = cont.detach().cpu().numpy().astype(np.float16)
            print(f"  frames {b}-{min(b + batch_size, T)}/{T}")

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            contact=contacts,
            times=np.asarray(times, dtype=np.float32),
            frame_idx=np.asarray(fidx, dtype=np.int32),
            bbox=np.asarray(bboxes, dtype=np.int32),
            had_person=np.asarray(had_person, dtype=bool),
            sample_fps=np.float32(sample_fps),
            video_fps=np.float32(video_fps),
            num_source_frames=np.int32(n_src),
            width=np.int32(width),
            height=np.int32(height),
            crop=np.bool_(crop),
        )
        out = buf.getvalue()
        print(f"Done. contact {contacts.shape}, mean p={contacts.mean():.4f}, "
              f"output {len(out)/1024:.1f} KB")
        return out


def estimate_contacts_deco(
    input_path: str,
    output_path: str | None = None,
    sample_fps: float = 5.0,
    crop: bool = True,
) -> str:
    """Client-side: send a local video to Modal DECO, save per-frame contacts.

    Returns the path to the saved .npz.
    """
    from pathlib import Path

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".deco.npz")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("DECO Contact Estimation (Modal L4)")
    print(f"  Input: {input_path}")
    print(f"  Sample fps: {sample_fps} | crop: {crop}")
    print(f"  Uploading {input_path.stat().st_size / 1024 / 1024:.1f} MB ...")

    with open(input_path, "rb") as f:
        video_bytes = f.read()

    estimator = modal.Cls.from_name("deco-contact-estimation", "DECOEstimator")()
    out_bytes = estimator.estimate_contacts.remote(
        video_bytes, sample_fps=sample_fps, crop=crop
    )

    with open(output_path, "wb") as f:
        f.write(out_bytes)

    print(f"  Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return str(output_path)


@app.local_entrypoint()
def main(video: str, out: str = "", sample_fps: float = 5.0, crop: bool = True):
    """`modal run` convenience entrypoint for a single video."""
    estimate_contacts_deco(video, out or None, sample_fps=sample_fps, crop=crop)
