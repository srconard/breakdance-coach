# CLAUDE.md - Project Context for Claude Code

**Last Updated:** February 24, 2026 (evening session)

## What Is This Project?

**Breakdance Coach** — A suite of AI-powered tools for learning breakdancing:

1. **Tutorial Wiki Generator** (`tutorial_generator/`) — Converts YouTube videos into step-by-step GIF/video tutorials with AI descriptions, outputting Obsidian-compatible Markdown
2. **3D Move Analyzer** (`analyzer_3d/`) — Generates interactive 3D models from breakdancing videos using pose estimation (GVHMR → SMPL → GLB)
3. **Shared Tools** (`shared/`) — Frame interpolation (RIFE + FFmpeg), video download, cloud GPU utilities

## Current Status: 🟢 Working

Tutorial generator works end-to-end. RIFE frame interpolation deployed on Modal.com (cloud GPU). Re-clip tool enables upgrading clip quality post-generation. 3D analyzer pipeline works end-to-end: GVHMR pose estimation on Modal (T4 GPU) producing full SMPL body mesh, Blender 4.4 GLB export with animated armature + skinned mesh. Coordinate transform (SMPL Y-up to Blender Z-up) recently added, pending visual verification.

### What's Working
- ✅ YouTube downloads (via yt-dlp with android_vr/tv client workaround)
- ✅ Multi-quality YouTube downloads (720p, 1080p, best)
- ✅ Video preprocessing (71% size reduction for cheap Gemini analysis)
- ✅ Gemini video analysis (step identification) using `gemini-2.5-flash`
- ✅ Description generation (with rate limiting) using `gemini-2.5-flash`
- ✅ GIF/MP4/WebM output (from original video)
- ✅ Obsidian-compatible markdown (with YouTube source URL)
- ✅ Tutorial metadata JSON export (timestamps, settings, source URL)
- ✅ Re-clip tool — re-extract clips at higher quality without re-running AI
- ✅ Frame interpolation — RIFE v4.25 on Modal.com (cloud T4 GPU) + FFmpeg fallback
- ✅ Project reorganized into monorepo (tutorial_generator/, analyzer_3d/, shared/)
- ✅ 3D analyzer pipeline code (GVHMR Modal + Blender GLB export + CLI)
- ✅ GitHub repo: https://github.com/srconard/breakdance-coach
- ✅ GVHMR pose estimation fully working on Modal (bypasses Hydra config with OmegaConf + hydra.utils.instantiate)
- ✅ Full SMPL body mesh extraction (6890 vertices, 13776 faces, 24-joint skinning weights) via smplx on Modal
- ✅ Real SMPL rest pose joint positions extracted for proper bone placement
- ✅ Blender 4.4 (x64) GLB export with animated armature + skinned mesh
- ✅ Coordinate transform: SMPL Y-up to Blender Z-up `(x, y, z) -> (x, -z, y)`
- ✅ Full 3D pipeline: Video -> GVHMR (Modal T4) -> SMPL params + mesh -> Blender GLB -> Obsidian markdown
- ✅ Uniform bone directions in armature — all bones point +Y so SMPL rotations apply correctly to all joints (fixes leg/spine mismatch)

### In Progress
- 🟡 Obsidian 3D viewer integration (have "3D Embed" plugin; may need "model-viewer" or custom HTML embed for animation playback)
- 🟡 DeepMotion API access (requested, waiting for approval)

### Open Tasks
- 🟡 Refine Gemini prompt to skip talking-head segments (`tutorial_generator/src/video_analyzer.py` lines 82-102)
- 📋 Custom Three.js 3D viewer (Phase 3)

---

## Quick Start

```bash
# Set API key
set GOOGLE_API_KEY=your_google_api_key_here

# === Tutorial Generator ===

# Run with MP4 output (recommended - small files, playback controls)
python -m tutorial_generator.src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# Download from YouTube directly
python -m tutorial_generator.src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# Re-clip a step at 1080p from YouTube
python -m tutorial_generator.src.reclip "output/My_Tutorial" --step 11 --download-hq 1080p

# Slow-mo a clip with RIFE (cloud GPU)
python -m shared.interpolate "output/tutorial/gifs/step_05.mp4" --slowdown 3 --backend rife

# Slow-mo with FFmpeg (local, no GPU needed)
python -m shared.interpolate "output/tutorial/gifs/step_05.mp4" --slowdown 3 --backend ffmpeg

# === 3D Move Analyzer ===

# Setup (one-time): download checkpoints + upload to Modal
python -m analyzer_3d.src.gvhmr_setup

# Deploy GVHMR to Modal
python -m modal deploy analyzer_3d/src/gvhmr_modal.py

# Analyze a video → 3D GLB model
python -m analyzer_3d.src.main "video.mp4" --backend gvhmr

# Analyze specific tutorial steps
python -m analyzer_3d.src.main "video.mp4" --backend gvhmr \
    --metadata "output/My_Tutorial/tutorial_metadata.json" --step 3 --step 5
```

## Project Architecture

### Monorepo Structure
```
breakdance-coach/
├── tutorial_generator/      # Tool 1: Tutorial Wiki Generator
│   ├── src/
│   │   ├── main.py          # CLI entry point
│   │   ├── video_analyzer.py
│   │   ├── description.py
│   │   ├── gif_creator.py
│   │   ├── output.py
│   │   ├── reclip.py
│   │   └── video_prep.py
│   ├── templates/
│   │   └── tutorial.md
│   └── config.py
├── analyzer_3d/             # Tool 2: 3D Move Analyzer
│   ├── src/
│   │   ├── main.py          # CLI entry point
│   │   ├── gvhmr_modal.py   # GVHMR on Modal.com (T4 GPU)
│   │   ├── gvhmr_setup.py   # Checkpoint download & upload
│   │   ├── exporter.py      # SMPL → GLB via Blender headless
│   │   ├── output.py        # Obsidian markdown generation
│   │   └── blender_scripts/
│   │       └── smpl_to_glb.py
│   └── templates/
│       └── 3d_tutorial.md
├── shared/                  # Shared utilities
│   ├── downloader.py        # YouTube download (yt-dlp)
│   ├── interpolate.py       # Frame interpolation (RIFE + FFmpeg)
│   └── rife_modal.py        # RIFE v4.25 on Modal.com
├── CLAUDE.md
├── README.md
└── requirements.txt
```

### Tutorial Generator Pipeline
```
Video Input (YouTube URL or local file)
    ↓
[yt-dlp] Download video (or skip with --local-file)
    ↓
[ffmpeg] Preprocess for API (downscale 480p, 15fps) — saves Gemini tokens
    ↓
[Gemini API] Analyze preprocessed video → identify steps with timestamps
    ↓
[Configurable LLM] Generate descriptions (Google/Anthropic/OpenAI)
    ↓
[ffmpeg] Create GIFs/MP4/WebM from ORIGINAL video (high quality)
    ↓
[Jinja2] Generate Obsidian Markdown + tutorial_metadata.json
```

### 3D Analyzer Pipeline
```
Video Input (local file or YouTube URL)
    ↓
[GVHMR on Modal T4 GPU] (app: gvhmr-pose-estimation)
    ├── YOLO: Person detection + tracking
    ├── ViTPose: 2D keypoint estimation
    ├── HMR2: Feature extraction
    ├── GVHMR: Gravity-aware 3D pose → SMPL parameters
    └── smplx: Extract body mesh (6890 verts, 13776 faces, 24-joint weights, rest pose joints)
    ↓ Returns: SMPL params + mesh data (.pkl)
    ↓
[Blender 4.4 headless] (local, x64)
    ├── Build armature from SMPL rest pose joints
    ├── Coordinate transform: SMPL Y-up → Blender Z-up: (x,y,z) → (x,-z,y)
    ├── Create skinned mesh with vertex groups + weights
    └── Animate via per-frame bone rotations → .GLB export
    ↓
[Jinja2] Generate Obsidian Markdown with ![[move.glb#autoplay]]
```

#### GVHMR Config Bypass (Critical)
The standard `register_store_gvhmr()` imports training-only dependencies that fail in inference.
The permanent fix uses `OmegaConf.create()` + `hydra.utils.instantiate()` to build the config manually.
See `analyzer_3d/src/gvhmr_modal.py` for implementation.

## Key Files

| File | Purpose |
|------|---------|
| `tutorial_generator/src/main.py` | Tutorial generator CLI entry point |
| `tutorial_generator/src/video_analyzer.py` | Gemini video analysis (uses `gemini-2.5-flash`) |
| `tutorial_generator/src/description.py` | Multi-provider LLM with rate limiting |
| `tutorial_generator/src/gif_creator.py` | GIF/MP4/WebM creation |
| `tutorial_generator/src/output.py` | Obsidian markdown + tutorial_metadata.json |
| `tutorial_generator/src/reclip.py` | Re-clip tool — extract HQ clips from metadata |
| `tutorial_generator/config.py` | API key via `GOOGLE_API_KEY` env var |
| `analyzer_3d/src/main.py` | 3D analyzer CLI entry point |
| `analyzer_3d/src/gvhmr_modal.py` | GVHMR deployed on Modal.com (T4 GPU) |
| `analyzer_3d/src/gvhmr_setup.py` | Checkpoint download & Modal volume upload |
| `analyzer_3d/src/exporter.py` | SMPL → GLB via Blender headless |
| `analyzer_3d/src/output.py` | 3D model Obsidian markdown generation |
| `shared/downloader.py` | YouTube download with quality presets (yt-dlp) |
| `shared/interpolate.py` | Frame interpolation (RIFE cloud + FFmpeg local) |
| `shared/rife_modal.py` | RIFE v4.25 deployed on Modal.com (T4 GPU) |

## Documentation

| File | Contents |
|------|----------|
| `AGENTS/PROJECT-ROADMAP.md` | Full roadmap, 3D feature spec, reorg plan |
| `AGENTS/frame-interpolation-options.md` | Research: RIFE, fal.ai, Replicate, Topaz, etc. |
| `AGENTS/session-log-2026-02-24.md` | Latest session (RIFE, re-clip, HQ downloads, metadata) |
| `AGENTS/session-log-2026-02-23.md` | YouTube fix, GitHub setup, FFmpeg interpolation |
| `AGENTS/session-log-2026-01-23.md` | Build & test history |
| `README.md` | User-facing documentation |

## Configuration

### API Keys
```bash
# Windows — required for tutorial generator
set GOOGLE_API_KEY=your_google_api_key_here

# For Modal.com (RIFE + GVHMR)
python -m modal setup   # One-time auth via browser

# For DeepMotion (when API access granted)
set DEEPMOTION_CLIENT_ID=your_client_id
set DEEPMOTION_CLIENT_SECRET=your_client_secret

# Linux/Mac
export GOOGLE_API_KEY=your_google_api_key_here
```

### Defaults
- Width: 640px
- FPS: 12
- Format: GIF (use `--format mp4` for smaller files with controls)
- Source: Original video (not preprocessed) for clips
- Gemini model: `gemini-2.5-flash`
- Interpolation backend: `ffmpeg` (use `--backend rife` for higher quality)
- 3D backend: `gvhmr` (gravity-aware, best for breakdancing)
