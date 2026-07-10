"""AFUN affordance-foundation-model inference on Modal cloud GPU — SEE-layer v0 pilot.

Runs the released AFUN checkpoint (github.com/EricWang12/AFUN, arXiv 2606.02551:
Qwen3-VL-8B + SAM3 + Depth-Anything-3 + Sonata) against wild terrain photos with
language task phrases ("swing from the tree branch"), producing task-conditional
functional masks + post-contact motion curves. Photos are RGB-only; AFUN's own
DA3 path synthesizes depth.

Part of the movement-simulation desk's SEE-layer go/no-go pilot (2026-07-10).

Usage:
    cd C:/dev/breakdance-coach
    python -m modal run analyzer_3d/src/afun_modal.py

Outputs (to output/see_pilot/):
    afun_results.json                — per photo x task: confidence, detections, mask stats
    {photo}_afun_{task}.png          — AFUN's own mask + motion-curve overlay
"""

import json
from pathlib import Path

import modal

app = modal.App("see-pilot-afun")

# Reuse the pilot's HF cache volume; AFUN checkpoints live beside it.
weights_volume = modal.Volume.from_name("see-pilot-hf-cache", create_if_missing=True)

# Task-conditional phrases (AFUN wants imperative interaction language).
TASKS = {
    "swing": "swing from the tree branch",
    "vault": "vault over the rock",
    "balance": "balance across the ledge",
    "footwork": "do dance steps on the flat ground",
    # CLB is a first-class P2 inventory class; trees dominate 4 of 6 photos.
    "climb": "climb the tree",
}

afun_image = (
    # devel base = nvcc available in case a dep needs a source build
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11"
    )
    .apt_install(
        "git", "wget", "libgl1", "libglib2.0-0", "libgomp1", "libegl1", "libx11-6",
    )
    .pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "torchaudio==2.10.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        "git clone --recursive https://github.com/EricWang12/AFUN.git /AFUN",
        "pip install -r /AFUN/requirements.txt",
    )
    .run_commands(
        # xformers wheel must match torch+CUDA (see AFUN install.sh)
        "pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 xformers==0.0.35",
        "pip install spconv-cu124",
        "pip install torch_scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html",
        "pip install -e /AFUN/src/models/decoder/sam3 --no-deps",
    )
    .env({
        "HF_HOME": "/weights/hf",
        "AFUN_ROOT": "/AFUN",
        # 23 GB A10 is tight for the 8B stack; avoid allocator fragmentation
        # (both names: torch <=2.9 / >=2.10)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })
)


@app.cls(
    image=afun_image,
    # Single A10 (23 GB): Qwen3-VL-8B (bf16 ~17 GB) + SAM3 + DA3 — tight but
    # the AFUN pipeline is single-device ("cuda:0"). L40S/A100 are
    # payment-method-gated on this workspace.
    gpu="A10G",
    timeout=3600,
    scaledown_window=120,
    volumes={"/weights": weights_volume},
)
class AFUNModel:
    @modal.enter()
    def setup(self):
        """Fetch checkpoints + patch the config. Model loading is deferred to
        predict_batch so DA3 depth (stage 1) and the 8B AFUN stack (stage 2)
        never co-reside on the 23 GB A10.
        """
        import os
        import subprocess
        import sys

        subprocess.run(["nvidia-smi"], check=False)

        os.chdir("/AFUN")
        sys.path.insert(0, "/AFUN")

        # Keep the big checkpoints on the persistent volume.
        ckpt_store = "/weights/afun_checkpoints"
        os.makedirs(ckpt_store, exist_ok=True)
        if not os.path.islink("/AFUN/checkpoints"):
            if os.path.isdir("/AFUN/checkpoints"):
                # merge any repo-shipped stubs, then replace with symlink
                for f in os.listdir("/AFUN/checkpoints"):
                    os.rename(f"/AFUN/checkpoints/{f}", f"{ckpt_store}/{f}")
                os.rmdir("/AFUN/checkpoints")
            os.symlink(ckpt_store, "/AFUN/checkpoints")

        # 1) afun.pt from Google Drive (repo default source).
        afun_pt = f"{ckpt_store}/afun.pt"
        if not os.path.exists(afun_pt):
            import gdown

            print("Downloading afun.pt from Google Drive...")
            gdown.download(id="1J4TNj_w3b7BTaeriBPuDJCn7q_SlV6PR", output=afun_pt, quiet=False)
            if not os.path.exists(afun_pt):
                raise RuntimeError("gdown did not produce afun.pt (Drive quota?)")
            weights_volume.commit()

        # 2) SAM3 checkpoint. facebook/sam3 is LICENSE-GATED on HF (401 without
        # an authorized token; this workspace has none). The public mirror
        # 1038lab/sam3 carries the same sam3.pt; build_sam3_image_model() with
        # an explicit checkpoint_path never touches the gated repo (its
        # config.json download is dead code — the architecture is built in
        # code). Safety: validate with torch.load(weights_only=True), which
        # rejects any pickled code — tensors only.
        sam3_pt = f"{ckpt_store}/sam3.pt"
        if not os.path.exists(sam3_pt):
            from huggingface_hub import hf_hub_download

            print("Downloading sam3.pt from public mirror 1038lab/sam3...")
            hf_hub_download(repo_id="1038lab/sam3", filename="sam3.pt", local_dir=ckpt_store)
            weights_volume.commit()

        import torch

        print("Validating sam3.pt (weights_only=True: tensors only, no code)...")
        ckpt = torch.load(sam3_pt, map_location="cpu", weights_only=True)
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)
        print(f"sam3.pt OK, top-level: {keys if not isinstance(keys, list) else keys[:5]}")
        del ckpt

        # 3) Point the decoder at the local SAM3 checkpoint (skip the gated fetch).
        import yaml

        cfg_path = "/AFUN/configs/inference.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("decoder", {}).setdefault("model_cfg", {})
        cfg["decoder"]["model_cfg"]["sam3_checkpoint_path"] = sam3_pt
        cfg["decoder"]["model_cfg"]["sam3_load_from_hf"] = False
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print("Patched configs/inference.yaml with local SAM3 checkpoint.")

        self.ckpt_store = ckpt_store
        self.model = None

    @modal.method()
    def predict_batch(self, photos: dict[str, bytes], tasks: dict[str, str]) -> dict:
        """Run every task phrase against every photo (RGB-only; DA3 depth).

        Returns:
            {photo: {task_key: {"score": float, "num_detections": int,
                                "mask_frac": float, "png": overlay bytes}}}
        """
        import gc
        import io
        import os
        import shutil
        import sys
        import tempfile

        import numpy as np
        import torch
        from PIL import Image

        os.chdir("/AFUN")
        if "/AFUN" not in sys.path:
            sys.path.insert(0, "/AFUN")

        def k_to_cam(K):
            K = np.asarray(K)
            return {
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            }

        # ---- Stage 1: DA3 depth for every photo, then unload DA3 (GIANT
        # backbone; cannot share the 23 GB A10 with the Qwen3-VL-8B stack) ----
        from depth import prepare_depth

        images: dict[str, Image.Image] = {}
        depths: dict[str, tuple] = {}
        for photo_name, photo_bytes in photos.items():
            rgb_pil = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            images[photo_name] = rgb_pil
            print(f"[{photo_name}] DA3 depth ({rgb_pil.width}x{rgb_pil.height})...")
            depth_mm, K, depth_source = prepare_depth(
                np.asarray(rgb_pil), None, device="cuda:0", refine=True
            )
            depths[photo_name] = (depth_mm, k_to_cam(K), depth_source)

        import depth.da3 as _da3_mod

        _da3_mod._DA3 = None  # release the DA3 model
        gc.collect()
        torch.cuda.empty_cache()
        weights_volume.commit()  # persist DA3 weights fetched on demand
        print("Stage 1 done: depths computed, DA3 unloaded.")

        # ---- Stage 2: load AFUN (Qwen3-VL-8B + SAM3 + Sonata) and infer ----
        from src.inference import infer_image, load_model, save_results

        if self.model is None:
            print("Loading AFUN model...")
            self.model = load_model(
                f"{self.ckpt_store}/afun.pt", "inference", device="cuda:0"
            )
            weights_volume.commit()  # persist backbones fetched at load
            print("AFUN loaded.")

        results = {}
        for photo_name, rgb_pil in images.items():
            depth_mm, cam, depth_source = depths[photo_name]
            results[photo_name] = {}
            for key, phrase in tasks.items():
                out_dir = tempfile.mkdtemp()
                result = None
                used_scale = 1.0
                try:
                    # 23 GB A10 leaves <1 GiB headroom over the loaded stack;
                    # some scenes' point clouds exceed it. On OOM, shrink
                    # image + depth + intrinsics together and retry.
                    import cv2

                    # Retry ladder: sky-heavy photos (B/D/F) blow up the
                    # depth point cloud regardless of image scale, so later
                    # rungs also CAP far depth (>cap mm -> 0 = invalid),
                    # which removes sky/far-canopy points.
                    last_oom = None
                    for scale, depth_cap in (
                        (1.0, None), (1.0, 6000.0), (0.7, 5000.0), (0.5, 4000.0),
                    ):
                        img, dep, camd = rgb_pil, np.asarray(depth_mm), cam
                        if scale < 1.0:
                            nw = max(64, int(rgb_pil.width * scale))
                            nh = max(64, int(rgb_pil.height * scale))
                            img = rgb_pil.resize((nw, nh), Image.LANCZOS)
                            dep = cv2.resize(dep, (nw, nh), interpolation=cv2.INTER_NEAREST)
                            camd = {k: v * scale for k, v in cam.items()}
                        if depth_cap is not None:
                            dep = np.where(dep > depth_cap, 0.0, dep).astype(dep.dtype)
                        try:
                            with torch.no_grad():
                                result = infer_image(
                                    self.model,
                                    image=img,
                                    query=phrase,
                                    device="cuda:0",
                                    depth=dep,
                                    cam=camd,
                                )
                            used_scale = scale
                            last_oom = None
                            break
                        except torch.cuda.OutOfMemoryError as oom:
                            last_oom = oom
                            gc.collect()
                            torch.cuda.empty_cache()
                            print(f"  {photo_name} / {key}: OOM at scale={scale} cap={depth_cap}, retrying...")
                    if last_oom is not None:
                        import traceback

                        print("  final OOM traceback (last frames):")
                        print("\n".join(traceback.format_exception(last_oom)[-6:]))
                        raise last_oom
                    save_results(result, out_dir, tag="pred")
                    entry = {
                        "task": phrase,
                        "depth_source": depth_source,
                        "scale": used_scale,
                        "score": float(result.get("score") or 0.0),
                        "num_detections": int(result.get("num_detections") or 0),
                    }
                    npz_path = os.path.join(out_dir, "pred.npz")
                    if os.path.exists(npz_path):
                        data = np.load(npz_path, allow_pickle=True)
                        if "mask" in data:
                            mask = data["mask"]
                            entry["mask_frac"] = float(
                                np.asarray(mask).astype(bool).mean()
                            )
                    png_path = os.path.join(out_dir, "pred_seg.png")
                    if os.path.exists(png_path):
                        with open(png_path, "rb") as f:
                            entry["png"] = f.read()
                    results[photo_name][key] = entry
                    print(
                        f"  {photo_name} / {key}: score={entry['score']:.3f} "
                        f"det={entry['num_detections']} "
                        f"mask={entry.get('mask_frac', -1):.4f}"
                    )
                except Exception as e:
                    results[photo_name][key] = {
                        "task": phrase,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    print(f"  {photo_name} / {key}: ERROR {e}")
                finally:
                    shutil.rmtree(out_dir, ignore_errors=True)
                    # 23 GB A10: free per-inference allocations or photo 2+ OOMs
                    del result
                    gc.collect()
                    torch.cuda.empty_cache()
        return results


# ---------------------------------------------------------------------------
# Local side
# ---------------------------------------------------------------------------

VAULT = Path("C:/Users/Shawn/Documents/Workspace/Shawn's Vault")
OUTPUT_DIR = Path("C:/dev/breakdance-coach/output/see_pilot")

# Same 6 photos as robopoint_modal.py (P2 affordance-report pilot, photo audit §1).
PHOTOS = {
    "A_creek_running": VAULT / "Settings/Locations/2025-05-23_google-photo_172516.jpg",
    "B_tree_station": VAULT / "Settings/Locations/2025-05-23_google-photo_180528.jpg",
    "C_manor_ruins_tree": VAULT / "Settings/Locations/2025-05-16_google-photo_171459.jpg",
    "D_cynwyd_rope_tree": VAULT / "Settings/Locations/2025-05-15_google-photo_142513.jpg",
    "E_west_laurel_woods": VAULT / "Settings/Locations/2025-05-15_google-photo_132911.jpg",
    "F_trail_multitrunk": VAULT / "Settings/Images/20241006_141451.jpg",
}


def photo_jpeg_bytes(photo_path: Path, max_dim: int = 640) -> bytes:
    """EXIF-normalized JPEG bytes (phone photos store rotation as EXIF tag 6;
    the model must see upright pixels). Downscaled to max_dim: AFUN's depth
    point cloud on a 23 GB A10 cannot take a 10 MP sky-heavy photo (constant
    8.4 GiB allocation for full-res F regardless of later rescaling)."""
    import io

    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
    if max(image.size) > max_dim:
        s = max_dim / max(image.size)
        image = image.resize(
            (round(image.width * s), round(image.height * s)), Image.LANCZOS
        )
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    photos = {}
    for name, path in PHOTOS.items():
        if path.exists():
            photos[name] = photo_jpeg_bytes(path)
        else:
            print(f"MISSING photo: {path}")
    print(f"Sending {len(photos)} photos x {len(TASKS)} tasks to Modal (L40S)...")

    results = AFUNModel().predict_batch.remote(photos, TASKS)

    # Merge over previous runs: a success is never overwritten by an error.
    results_path = OUTPUT_DIR / "afun_results.json"
    summary = json.loads(results_path.read_text()) if results_path.exists() else {}
    for photo_name, tasks in results.items():
        summary.setdefault(photo_name, {})
        for key, entry in tasks.items():
            png = entry.pop("png", None)
            if png:
                out_png = OUTPUT_DIR / f"{photo_name}_afun_{key}.png"
                out_png.write_bytes(png)
                entry["overlay"] = out_png.name
                print(f"  overlay -> {out_png}")
            if "error" in entry and "error" not in summary[photo_name].get(key, {"error": 1}):
                continue  # keep the earlier success
            summary[photo_name][key] = entry

    results_path.write_text(json.dumps(summary, indent=2))
    print(f"Results -> {results_path}")
    print("Done.")
