# Frame Interpolation Options Research

**Date:** February 23, 2026
**Status:** Research complete. Ready to implement when needed.
**Current tool:** `src/interpolate.py` uses FFmpeg minterpolate (basic, artifacts on fast motion)

---

## Problem

FFmpeg's `minterpolate` filter uses basic optical flow (block matching). It works OK for slow/simple motion but produces **ghosting and smearing** on fast, complex movements like breakdancing (especially flares, V-sit circles, power moves). We need better options.

---

## Options Comparison

| Tool | Quality (Dance) | Speed | Cost | CLI/Python | Setup |
|------|----------------|-------|------|------------|-------|
| **FFmpeg minterpolate** | Poor | Fast | Free | Yes | Already installed |
| **RIFE (Practical-RIFE)** | Good-Excellent | Very fast | Free (MIT) | Yes | NVIDIA GPU + pip |
| **RIFE-NCNN-Vulkan** | Good-Excellent | Fast | Free | Yes (binary) | Any GPU, prebuilt binary |
| **fal.ai (FILM API)** | Good | Cloud | ~$0.05-0.20/video | Yes (pip) | API key only |
| **Replicate (ST-MFNet)** | Good | Cloud | ~$0.30/video | Yes (pip) | API key only |
| **Replicate (FILM)** | Good | Cloud | ~$0.13/video | Yes (pip) | API key only |
| **Modal (self-host RIFE)** | Excellent | Cloud | ~$0.005/video | Yes | Write wrapper |
| **Topaz Apollo API** | Best (dance-specific) | Slow | $50+/mo | Yes (topyaz) | Subscription |
| **FILM (Google)** | Very good (large motion) | Slow | Free | Yes | ARCHIVED, TF 2.6 |

---

## Recommended Path

### Option A: Cloud API (Easiest, Good Quality)

**fal.ai** or **Replicate** — pip install, set API key, call from Python.

```python
# fal.ai example
pip install fal-client
import fal_client
result = fal_client.run("fal-ai/film/video", arguments={
    "video_url": "https://example.com/clip.mp4",
    "num_frames": 2,               # frames to generate between originals
    "use_scene_detection": True,    # avoids smear at scene cuts
    "video_quality": "high",
    "fps": 60
})
```

```python
# Replicate example
pip install replicate
import replicate
output = replicate.run("zsxkib/st-mfnet", input={
    "mp4": open("clip.mp4", "rb"),
    # supports up to 16x frame doubling
})
```

**Pros:** No GPU needed, instant setup, good quality
**Cons:** Per-video cost, requires internet, upload/download time

### Option B: Local RIFE (Best Quality, Free, Needs NVIDIA GPU)

**Practical-RIFE** — If this machine has an NVIDIA GPU with 6+ GB VRAM.

```bash
git clone https://github.com/hzwer/Practical-RIFE
cd Practical-RIFE
pip install torch torchvision
python inference_video.py --multi=2 --video=clip.mp4        # 2x slowmo
python inference_video.py --multi=2 --video=clip.mp4 --fp16 # faster
```

**Tip:** Run 2x interpolation twice (iteratively) rather than 4x in one pass — much less ghosting on fast moves.

**Pros:** Best quality, free, fast on GPU, no internet needed
**Cons:** Requires NVIDIA GPU, PyTorch install

### Option C: Self-Host RIFE on Cloud GPU (Best Quality, Cheapest at Scale)

**Modal.com** — Write a Python function, deploy to cloud GPU, call from CLI.

- T4 GPU: $0.59/hr → processing a 13s clip in ~30s costs **< $0.01**
- $30/mo free credits (Starter plan)
- Scales to zero (no idle charges)
- Python-native (no Docker/YAML)

**Pros:** RIFE quality, very cheap, no local GPU needed
**Cons:** Initial setup (~50-100 lines wrapper code), requires internet

### Option D: Topaz Apollo (Highest Quality, Expensive)

**Topaz Video AI** with Apollo model — specifically trained for dance/sports.

```bash
pip install topyaz
topyaz video ./clip.mp4 --model prob-3 --interpolate --fps 60 --output ./slowmo.mp4
```

Or via their REST API ($50/mo Developer plan, $0.10/credit).

**Pros:** Absolute best quality for dance, up to 8x slowdown
**Cons:** $50+/mo subscription, slower processing

---

## Key Technical Notes

- **RIFE v4.25/v4.26** is the community standard — MIT license, 50-100+ fps on modern GPUs
- **SG-RIFE** (Dec 2025 paper) adds semantic guidance for even better quality — watch for public release
- **FILM** is archived (Oct 2025) but still works — hosted versions on fal.ai/Replicate are the easiest way to use it
- **Iterative 2x is better than single 4x** — run 2x twice for less ghosting on fast moves
- **minterpolate modes matter** — `mi_mode=mci` + `mc_mode=aobmc` is already the best FFmpeg can do

---

## Integration Plan

When ready to implement, update `src/interpolate.py` to support multiple backends:

```python
python -m src.interpolate "clip.mp4" --slowdown 3 --backend ffmpeg     # current (basic)
python -m src.interpolate "clip.mp4" --slowdown 3 --backend rife       # local RIFE
python -m src.interpolate "clip.mp4" --slowdown 3 --backend fal        # fal.ai cloud
python -m src.interpolate "clip.mp4" --slowdown 3 --backend replicate  # Replicate cloud
```

Each backend would implement a common interface. FFmpeg stays as the zero-dependency fallback.
