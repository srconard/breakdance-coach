# CLAUDE.md - Project Context for Claude Code

**Last Updated:** February 23, 2026

## What Is This Project?

**Breakdance Coach** — A suite of AI-powered tools for learning breakdancing:

1. **Tutorial Wiki Generator** (`src/`) — Converts YouTube videos into step-by-step GIF/video tutorials with AI descriptions, outputting Obsidian-compatible Markdown
2. **3D Move Analyzer** (planned) — Generates interactive 3D models from breakdancing videos using pose estimation (GVHMR/PromptHMR → SMPL → GLB)
3. **Shared Tools** — Frame interpolation, video download, cloud GPU utilities

## Current Status: 🟢 Working

Tutorial generator works end-to-end. 3D analyzer is planned. Frame interpolation has basic FFmpeg backend with better options researched.

### What's Working
- ✅ YouTube downloads (via yt-dlp with android_vr/tv client workaround)
- ✅ Video preprocessing (71% size reduction)
- ✅ Gemini video analysis (step identification) using `gemini-2.5-flash`
- ✅ Description generation (with rate limiting) using `gemini-2.5-flash`
- ✅ GIF/MP4/WebM output (from original video)
- ✅ Obsidian-compatible markdown
- ✅ Frame interpolation tool (`src/interpolate.py`) — FFmpeg minterpolate
- ✅ GitHub repo: https://github.com/srconard/breakdance-coach

### Planned
- 📋 3D Move Analyzer (video → interactive 3D model in Obsidian)
- 📋 Upgraded frame interpolation (RIFE or cloud API — see `AGENTS/frame-interpolation-options.md`)
- 📋 Project reorganization into `tutorial-generator/`, `3d-analyzer/`, `shared/`

### Open Tasks
- 🟡 Refine Gemini prompt to skip talking-head segments (`src/video_analyzer.py` lines 82-102)
- 🟡 Upgrade frame interpolation quality (research done, implementation pending)

---

## Quick Start

```bash
# Set API key
set GOOGLE_API_KEY=your_google_api_key_here

# Run with MP4 output (recommended - small files, playback controls)
python -m src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# Download from YouTube directly
python -m src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# Slow-mo a specific clip
python -m src.interpolate "output/tutorial/gifs/step_05.mp4" --slowdown 3
```

## Project Architecture

### Tutorial Generator Pipeline
```
Video Input (YouTube URL or local file)
    ↓
[yt-dlp] Download video (or skip with --local-file)
    ↓
[ffmpeg] Preprocess for API (downscale 480p, 15fps) - saves tokens
    ↓
[Gemini API] Analyze preprocessed video → identify steps with timestamps
    ↓
[Configurable LLM] Generate descriptions (Google/Anthropic/OpenAI)
    ↓
[ffmpeg] Create GIFs/MP4/WebM from ORIGINAL video (high quality)
    ↓
[Jinja2] Generate Obsidian Markdown with embedded media
```

### 3D Analyzer Pipeline (Planned)
```
Video Input
    ↓
[YOLO + ViTPose] Person detection + 2D keypoints
    ↓
[GVHMR / PromptHMR] 3D mesh recovery → SMPL parameters
    ↓
[smplx] SMPL params → 3D mesh vertices (6,890 per frame)
    ↓
[Blender headless] Rig mesh, apply animation → .GLB export
    ↓
[Obsidian model-viewer] Interactive 3D viewer: ![[move.glb#autoplay]]
```

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | Tutorial generator CLI entry point |
| `src/video_analyzer.py` | Gemini video analysis (uses `gemini-2.5-flash`) |
| `src/downloader.py` | YouTube download (yt-dlp with android_vr/tv clients) |
| `src/description.py` | Multi-provider LLM with rate limiting (uses `gemini-2.5-flash`) |
| `src/gif_creator.py` | GIF/MP4/WebM creation |
| `src/output.py` | Obsidian markdown generation |
| `src/interpolate.py` | Frame interpolation tool (FFmpeg, upgrades planned) |
| `config.py` | API key via `GOOGLE_API_KEY` env var |

## Documentation

| File | Contents |
|------|----------|
| `AGENTS/PROJECT-ROADMAP.md` | Full roadmap, 3D feature spec, reorg plan |
| `AGENTS/frame-interpolation-options.md` | Research: RIFE, fal.ai, Replicate, Topaz, etc. |
| `AGENTS/session-log-2026-02-23.md` | Latest session (YouTube fix, GitHub, interpolation) |
| `AGENTS/session-log-2026-01-23.md` | Build & test history |
| `README.md` | User-facing documentation |

## Configuration

### API Key
```bash
# Windows
set GOOGLE_API_KEY=your_google_api_key_here

# Linux/Mac
export GOOGLE_API_KEY=your_google_api_key_here
```

### Defaults
- Width: 640px
- FPS: 12
- Format: GIF (use `--format mp4` for smaller files with controls)
- Source: Original video (not preprocessed)
- Gemini model: `gemini-2.5-flash`
