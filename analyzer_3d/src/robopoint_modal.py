"""RoboPoint spatial-affordance inference on Modal cloud GPU — SEE-layer v0 pilot.

Runs the released RoboPoint checkpoint (wentao-yuan/robopoint-v1-vicuna-v1.5-13b,
a 13B Vicuna-based LLaVA-style VLM, CoRL 2024) against wild terrain photos with
move-verb affordance queries ("point to tree limbs a person could swing from"),
parses the predicted 2D keypoints, and renders colored per-query dot overlays.

Part of the movement-simulation desk's SEE-layer go/no-go pilot (2026-07-10):
do released affordance models survive wild geometry (creek rock, bank trees)?

Usage:
    cd C:/dev/breakdance-coach
    python -m modal run analyzer_3d/src/robopoint_modal.py

Outputs (to output/see_pilot/):
    robopoint_results.json          — raw model text + parsed points per photo x query
    {photo}_robopoint.png           — combined overlay, one dot color per query
"""

import json
from pathlib import Path

import modal

app = modal.App("see-pilot-robopoint")

# HF cache on a persistent volume so the 26 GB checkpoint downloads once.
hf_cache = modal.Volume.from_name("see-pilot-hf-cache", create_if_missing=True)

MODEL_PATH = "wentao-yuan/robopoint-v1-vicuna-v1.5-13b"

# Official instruction suffix from robopoint/serve/gradio_web_server.py —
# this is what makes the model emit "[(x1, y1), ...]" normalized tuples.
POINT_SUFFIX = (
    " Your answer should be formatted as a list of tuples, i.e. "
    "[(x1, y1), (x2, y2), ...], where each tuple contains the x and y "
    "coordinates of a point satisfying the conditions above. The coordinates "
    "should be between 0 and 1, indicating the normalized pixel locations of "
    "the points in the image."
)

# Move-verb affordance queries (SEE-layer vocabulary) -> overlay color.
QUERIES = {
    "footwork": (
        "Find spots on flat surfaces where a person could do footwork or dance steps.",
        (255, 0, 255),   # magenta
    ),
    "vault": (
        "Find rocks or obstacles that a person could vault over.",
        (255, 140, 0),   # orange
    ),
    "swing": (
        "Find tree limbs or branches that a person could swing from.",
        (0, 255, 255),   # cyan
    ),
    "balance": (
        "Find ledges or narrow surfaces that a person could balance along.",
        (255, 255, 0),   # yellow
    ),
    "landing": (
        "Find good landing zones where a person could safely land after a jump.",
        (0, 255, 0),     # lime
    ),
    # Not in the mandated 5, but CLB is a first-class P2 inventory class and
    # trees dominate 4 of the 6 photos — needed for the found/missed scorecard.
    "climb": (
        "Find trees or rock faces that a person could climb.",
        (30, 100, 255),  # blue
    ),
}

robopoint_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("git+https://github.com/wentaoyuan/RoboPoint.git")
    # torch 2.1.2 (pinned by RoboPoint) predates numpy 2 — force numpy 1.x last.
    .pip_install("numpy==1.26.4")
    .env({"HF_HOME": "/hf_cache"})
)


@app.cls(
    image=robopoint_image,
    # 2x A10 (23 GB each): the 13B fp16 checkpoint (~27 GB) shards across both
    # via device_map="auto". (L40S/A100 are payment-method-gated on this
    # workspace; A10G is the largest free-tier GPU.)
    gpu="A10G:2",
    timeout=3600,
    scaledown_window=120,
    volumes={"/hf_cache": hf_cache},
)
class RoboPointModel:
    @modal.enter()
    def load(self):
        import torch
        from robopoint.model.builder import load_pretrained_model
        from robopoint.mm_utils import get_model_name_from_path
        from robopoint.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path(MODEL_PATH)
        print(f"Loading {MODEL_PATH} ({model_name})...")
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(MODEL_PATH, None, model_name)
        )
        hf_cache.commit()  # persist freshly downloaded weights
        self.device = torch.device("cuda")
        print("RoboPoint loaded.")

    @modal.method()
    def predict_batch(self, photos: dict[str, bytes], queries: dict[str, str]) -> dict:
        """Run every query against every photo.

        Args:
            photos: {photo_name: jpeg_bytes}
            queries: {query_key: query_text (without point suffix)}

        Returns:
            {photo_name: {query_key: {"raw": str, "points": [[x, y], ...]}}}
            with points in normalized [0, 1] image coordinates.
        """
        import io
        import re

        import torch
        from PIL import Image
        from robopoint.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from robopoint.conversation import conv_templates
        from robopoint.mm_utils import process_images, tokenizer_image_token

        tuple_re = re.compile(r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)")

        def parse_points(text):
            points = []
            for match in tuple_re.findall(text):
                try:
                    vals = [float(v) for v in match.split(",")]
                except ValueError:
                    continue
                if len(vals) == 2:
                    points.append(vals)
                elif len(vals) == 4:  # bbox -> center point
                    points.append([(vals[0] + vals[2]) / 2, (vals[1] + vals[3]) / 2])
            return points

        results = {}
        for photo_name, photo_bytes in photos.items():
            image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            images_tensor = process_images(
                [image], self.image_processor, self.model.config
            ).to(self.model.device, dtype=torch.float16)
            image_sizes = [image.size]

            results[photo_name] = {}
            for key, query in queries.items():
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query + POINT_SUFFIX
                conv = conv_templates["vicuna_v1"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = (
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    .unsqueeze(0)
                    .cuda()
                )
                try:
                    with torch.inference_mode():
                        output_ids = self.model.generate(
                            input_ids,
                            images=images_tensor,
                            image_sizes=image_sizes,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=512,
                            use_cache=True,
                        )
                    raw = self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )[0].strip()
                    results[photo_name][key] = {
                        "raw": raw,
                        "points": parse_points(raw),
                    }
                except Exception as e:  # keep the batch alive on a single failure
                    results[photo_name][key] = {"error": f"{type(e).__name__}: {e}"}
                n = len(results[photo_name][key].get("points", []))
                print(f"  {photo_name} / {key}: {n} points")
        return results


# ---------------------------------------------------------------------------
# Local side: photo set, overlay rendering, entrypoint
# ---------------------------------------------------------------------------

VAULT = Path("C:/Users/Shawn/Documents/Workspace/Shawn's Vault")
OUTPUT_DIR = Path("C:/dev/breakdance-coach/output/see_pilot")

# The 6 photos from the P2 affordance-report pilot (2026-07-09), photo audit §1.
PHOTOS = {
    "A_creek_running": VAULT / "Settings/Locations/2025-05-23_google-photo_172516.jpg",
    "B_tree_station": VAULT / "Settings/Locations/2025-05-23_google-photo_180528.jpg",
    "C_manor_ruins_tree": VAULT / "Settings/Locations/2025-05-16_google-photo_171459.jpg",
    "D_cynwyd_rope_tree": VAULT / "Settings/Locations/2025-05-15_google-photo_142513.jpg",
    "E_west_laurel_woods": VAULT / "Settings/Locations/2025-05-15_google-photo_132911.jpg",
    "F_trail_multitrunk": VAULT / "Settings/Images/20241006_141451.jpg",
}


def load_photo_normalized(photo_path: Path):
    """Load a photo with EXIF orientation applied (phone photos store rotation
    as EXIF tag 6; models must see upright pixels)."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")


def photo_jpeg_bytes(photo_path: Path) -> bytes:
    """EXIF-normalized JPEG bytes to send to the model."""
    import io

    image = load_photo_normalized(photo_path)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def render_overlay(photo_path: Path, photo_results: dict, out_path: Path) -> None:
    """Draw each query's points as colored dots + legend onto the photo."""
    from PIL import Image, ImageDraw

    image = load_photo_normalized(photo_path)
    # Upscale small thumbnails so dots and legend stay legible.
    if max(image.size) < 1024:
        scale = 1024 / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.LANCZOS,
        )
    draw = ImageDraw.Draw(image)
    W, H = image.size
    r = max(5, W // 130)

    for key, (_, color) in QUERIES.items():
        entry = photo_results.get(key, {})
        for x, y in entry.get("points", []):
            # Normalized coords expected; tolerate pixel coords.
            px = x * W if x <= 1.0 else x
            py = y * H if y <= 1.0 else y
            draw.ellipse(
                [px - r, py - r, px + r, py + r],
                fill=color,
                outline=(0, 0, 0),
                width=2,
            )

    # Legend strip (bottom).
    legend_h = 26 * len(QUERIES) + 10
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    legend = Image.new("RGB", (W, legend_h), (20, 20, 20))
    ldraw = ImageDraw.Draw(legend)
    y = 6
    for key, (query, color) in QUERIES.items():
        n = len(photo_results.get(key, {}).get("points", []))
        ldraw.ellipse([8, y + 3, 24, y + 19], fill=color, outline=(0, 0, 0))
        ldraw.text((32, y), f"{key} ({n}): {query}", fill=(230, 230, 230), font=font)
        y += 26
    combined = Image.new("RGB", (W, H + legend_h))
    combined.paste(image, (0, 0))
    combined.paste(legend, (0, H))
    combined.save(out_path)
    print(f"  overlay -> {out_path}")


@app.local_entrypoint()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    photos = {}
    for name, path in PHOTOS.items():
        if path.exists():
            photos[name] = photo_jpeg_bytes(path)
        else:
            print(f"MISSING photo: {path}")
    print(f"Sending {len(photos)} photos x {len(QUERIES)} queries to Modal (L40S)...")

    queries = {k: q for k, (q, _) in QUERIES.items()}
    results = RoboPointModel().predict_batch.remote(photos, queries)

    results_path = OUTPUT_DIR / "robopoint_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Raw results -> {results_path}")

    for name in photos:
        render_overlay(
            PHOTOS[name], results.get(name, {}), OUTPUT_DIR / f"{name}_robopoint.png"
        )
    print("Done.")
