# Breakdance Coach - Project Roadmap

**Last Updated:** February 24, 2026

---

## Vision

**Breakdance Coach** is a suite of AI-powered tools for learning breakdancing:

1. **Tutorial Wiki Generator** — Convert YouTube videos into step-by-step visual tutorials (Obsidian markdown)
2. **3D Move Analyzer** — Generate interactive 3D models from breakdancing videos
3. **Shared Tools** — Frame interpolation, video processing, cloud GPU utilities

---

## Project Structure (Planned Reorganization)

```
breakdance-coach/
├── tutorial-generator/          # Tool 1: Video → Step-by-step tutorials
│   ├── src/
│   │   ├── main.py              # CLI entry point
│   │   ├── video_analyzer.py    # Gemini video analysis
│   │   ├── description.py       # Multi-provider LLM descriptions
│   │   ├── downloader.py        # YouTube download (multi-quality)
│   │   ├── gif_creator.py       # GIF/MP4/WebM creation
│   │   ├── output.py            # Obsidian markdown + metadata JSON
│   │   ├── reclip.py            # Re-clip tool (HQ extraction)
│   │   └── video_prep.py        # Video preprocessing
│   ├── templates/
│   │   └── tutorial.md
│   └── config.py
│
├── 3d-analyzer/                 # Tool 2: Video → Interactive 3D models
│   ├── src/
│   │   ├── main.py              # CLI entry point
│   │   ├── pose_estimator.py    # 3D pose from video (GVHMR/PromptHMR)
│   │   ├── mesh_generator.py    # SMPL params → animated mesh
│   │   ├── exporter.py          # Export to .GLB/.FBX/.BVH
│   │   └── viewer.py            # Generate standalone HTML viewer
│   └── config.py
│
├── shared/                      # Shared utilities
│   ├── interpolate.py           # Frame interpolation (RIFE + FFmpeg)
│   ├── rife_modal.py            # RIFE v4.25 on Modal.com (cloud GPU)
│   ├── downloader.py            # YouTube download
│   └── gpu_cloud.py             # Modal/Replicate/fal.ai wrappers
│
├── output/                      # Generated content (gitignored)
├── downloads/                   # Downloaded videos (gitignored)
├── AGENTS/                      # Project docs and session logs
├── CLAUDE.md                    # Claude Code context
├── README.md
└── requirements.txt
```

**Rationale:** The tutorial generator and 3D analyzer are distinct tools with different dependencies (Gemini API vs PyTorch/SMPL). Shared utilities (interpolation, download, cloud GPU) are used by both. Each tool has its own CLI entry point so they can run independently.

---

## Current Status

### Tutorial Wiki Generator: 🟢 Working
| Component | Status |
|-----------|--------|
| YouTube downloads | ✅ Working (android_vr/tv workaround) |
| Multi-quality downloads | ✅ 720p, 1080p, best presets |
| Video preprocessing | ✅ 71% size reduction (for cheap Gemini analysis) |
| Gemini video analysis | ✅ `gemini-2.5-flash` |
| Description generation | ✅ `gemini-2.5-flash`, rate limited |
| GIF/MP4/WebM output | ✅ From original video |
| Obsidian markdown | ✅ With YouTube source URL |
| Metadata JSON export | ✅ Timestamps, settings, source URL |
| Re-clip tool | ✅ Re-extract at native quality, optional HQ download |
| Frame interpolation (FFmpeg) | ✅ Basic quality, local |
| Frame interpolation (RIFE) | ✅ High quality, Modal.com cloud GPU (T4) |

### 3D Move Analyzer: 📋 Planned
See detailed feature spec below.

### GitHub: https://github.com/srconard/breakdance-coach

---

## Tutorials Generated

| Tutorial | Source | Steps | Format |
|----------|--------|-------|--------|
| Flare Tutorial by Sambo | Local file (166MB) | 14 | GIF, MP4 |
| FLARE Workout Exercises | YouTube download (9.8MB) | 18 | MP4 |

---

## Open Tasks

### 1. 🟡 Refine Gemini Prompt for Better Step Detection
**Priority:** Medium | **Tool:** Tutorial Generator

The prompt identifies all sections including talking-head segments. Goal: only extract steps with actual move demonstrations.

**File:** `src/video_analyzer.py` (lines 82-102)

### 2. 🔵 Project Reorganization
**Priority:** Low | **Tool:** All

Reorganize from flat `src/` into `tutorial-generator/`, `3d-analyzer/`, `shared/` structure. Do this when starting the 3D analyzer to avoid disrupting working code.

---

## Completed Tasks

### ✅ RIFE Frame Interpolation on Modal.com (Feb 24, 2026)
- Deployed RIFE v4.25 (Practical-RIFE) on Modal.com with T4 GPU
- Model weights from Hugging Face mirror (r3gm/RIFE)
- Proper slow-motion output (writes at original fps, not multiplied fps)
- `src/rife_modal.py` — Modal app definition + client-side wrapper
- `src/interpolate.py` — `--backend rife` flag alongside `--backend ffmpeg`
- Tested on 1080p clip: 1920x1080, 1135 frames, 46.1 MB

### ✅ Re-clip Tool + Metadata JSON (Feb 24, 2026)
- `src/reclip.py` — Standalone CLI tool for re-extracting clips
- `src/output.py` — Saves `tutorial_metadata.json` alongside markdown
- Supports `--download-hq` to fetch higher quality from YouTube
- Quality-tagged filenames prevent overwriting existing downloads

### ✅ Multi-Quality YouTube Downloads (Feb 24, 2026)
- `src/downloader.py` — `quality` parameter with presets: `best_mp4`, `720p`, `1080p`, `best`
- Separate video+audio stream download with mp4 merge

### ✅ YouTube Source URL in Markdown (Feb 24, 2026)
- `templates/tutorial.md` — Source URL as blockquote
- `src/main.py` — `--source-url` CLI flag for use with `--local-file`

---

## Feature Spec: 3D Move Analyzer

### Overview

Generate an interactive 3D model/animation from a breakdancing video. The user can rotate, play/pause, change speed, and study the move from any angle.

### Why This Matters

- Breakdancing moves happen fast — a 3D model lets you pause and rotate to see exact body positioning
- Freezes and inversions are hard to understand from a single camera angle
- A 3D model shows the spatial relationships between limbs that flat video can't

### Technical Pipeline

```
Video File (breakdancing.mp4)
    │
    ▼
[1. Person Detection] ──── YOLO + ViTPose (2D keypoints)
    │
    ▼
[2. 3D Mesh Recovery] ──── GVHMR or PromptHMR
    │                       Output: per-frame SMPL params (.pkl/.npz)
    │                       - 10 shape params (body proportions)
    │                       - 72 pose params (24 joints × 3 rotation)
    │                       - 6,890 mesh vertices per frame
    ▼
[3. Mesh Generation] ───── smplx Python library
    │                       Converts SMPL params → 3D mesh vertices
    ▼
[4. Animation Export] ──── Blender headless scripting
    │                       Rigs mesh, applies per-frame pose
    │                       Output: animated .GLB file
    ▼
[5. Viewing] ──────────── Obsidian: obsidian-model-viewer plugin
                           Web: Three.js custom viewer or <model-viewer>
                           Desktop: Blender
```

### 3D Pose Estimation Options

| Method | Quality (Breakdancing) | Output | Status |
|--------|----------------------|--------|--------|
| **GVHMR** (SIGGRAPH Asia 2024) | Excellent — gravity-aware, handles inversions | SMPL params | Active, open source |
| **PromptHMR** (CVPR 2025) | Excellent — accepts pose/language prompts | SMPL params | Active, open source |
| **WHAM** (CVPR 2024) | Good — world-grounded but error accumulation | SMPL params | Active |
| **4D-Humans / HMR2.0** (ICCV 2023) | Good baseline — camera-space only | SMPL params, .obj | Active |
| **Meshcapade MoCapade** | Excellent (built on PromptHMR) | GLB, FBX, SMPL | Commercial API |
| **DeepMotion Animate 3D** | Good | FBX, BVH | Commercial API |

**Best bet for breakdancing:** GVHMR — handles gravity correctly (critical for freezes and inversions). PromptHMR is newer and can accept language descriptions of unusual poses.

**Breakdancing-specific challenge:** The BRACE dataset (ECCV 2022) was created specifically for breakdance pose estimation from Red Bull BC One videos. Acrobatic poses, motion blur, and inversions are known hard problems.

### SMPL Body Model

All modern methods output **SMPL parameters**:
- **10 shape betas** — body proportions (height, weight, limb lengths)
- **72 pose thetas** — rotation of 24 joints (3 Rodrigues rotation values each)
- **6,890 vertices** — fixed quad mesh topology, same for all people
- **SMPL-X** extends to 10,475 vertices including hands and face

From SMPL params, the `smplx` Python library reconstructs the full 3D mesh at any frame.

### Output Format: Animated GLB

**GLB (binary glTF)** is the target format because:
- Universal web standard for 3D
- Embeds mesh, skeleton, animation, materials in a single file
- Supported by Obsidian's model-viewer plugin
- Supported by Three.js, `<model-viewer>`, Blender, Unity

### Viewing in Obsidian

**Plugin:** [obsidian-model-viewer](https://github.com/janispritzkau/obsidian-model-viewer) uses Google's `<model-viewer>` web component.

```markdown
![[flare_move.glb#autoplay]]
![[flare_move.glb#height=400&autoplay]]
```

Supports: camera orbit, zoom, animation autoplay. Works with existing Obsidian workflow.

### Custom Three.js Viewer (Enhanced)

For full control beyond what `<model-viewer>` offers:

- Play/pause/scrub timeline
- Playback speed slider (0.1x to 2x)
- Camera orbit controls (rotate, zoom, pan)
- Toggle: wireframe / solid / skeleton overlay
- Side-by-side: original video + 3D model synced
- Frame stepping (←/→ arrow keys)
- Angle presets: front, side, top, custom

This would be a standalone HTML page generated alongside the tutorial markdown.

### GPU / Cloud Compute Strategy

These ML models require GPU inference. Options:

| Platform | GPU | Cost/hr | Best For |
|----------|-----|---------|----------|
| **Meshcapade API** | Managed | 10-99 EUR/mo | Quickest to integrate, no GPU setup |
| **DeepMotion API** | Managed | Freemium | Alternative commercial option |
| **Modal.com** | T4-A10G | $0.59-1.10/hr | Self-host GVHMR, cheapest at scale |
| **Replicate** | Managed | Per-run | If models get hosted there |
| **Local GPU** | Own | Free | If NVIDIA GPU with 8+ GB VRAM available |

**Recommended approach:**
1. **Phase 1:** Use Meshcapade or DeepMotion API for quick proof of concept
2. **Phase 2:** Self-host GVHMR on Modal.com for best quality and lowest cost at scale

### Implementation Phases

**Phase 1: Proof of Concept** (Commercial API)
- [ ] Sign up for Meshcapade or DeepMotion
- [ ] Upload a tutorial clip, get .GLB back
- [ ] Install obsidian-model-viewer plugin
- [ ] Embed in markdown: `![[move.glb#autoplay]]`
- [ ] Validate the concept works end-to-end

**Phase 2: Self-Hosted Pipeline** (Open Source)
- [ ] Deploy GVHMR on Modal.com (or local GPU)
- [ ] Write `pose_estimator.py` — video → SMPL params
- [ ] Write `mesh_generator.py` — SMPL params → mesh vertices via `smplx`
- [ ] Write `exporter.py` — mesh → animated .GLB via Blender headless
- [ ] Write CLI entry point
- [ ] Integrate with tutorial generator (optional: auto-generate 3D for each step)

**Phase 3: Custom Viewer**
- [ ] Build Three.js HTML viewer with full controls
- [ ] Sync video playback with 3D animation
- [ ] Add angle presets and skeleton overlay
- [ ] Generate viewer HTML alongside tutorial markdown

---

## Other Future Enhancements

### Tutorial Generator
- [ ] Use YouTube transcript as supplementary context
- [ ] Manual timestamp override option
- [ ] Review/edit steps before generation
- [ ] HTML output with embedded video player
- [ ] Configurable text overlays on clips

### User Experience
- [ ] Web interface (Streamlit/Gradio)
- [ ] Progress bar with ETA
- [ ] Preview mode
- [ ] Batch processing

---

## Technical Debt

1. Migrate from `google.generativeai` to `google.genai`
2. Add proper retry logic with exponential backoff
3. Add unit tests
4. Remove hardcoded Deno path (use PATH discovery)
5. Add input validation for URLs
6. Project reorganization (flat `src/` → monorepo with `tutorial-generator/`, `3d-analyzer/`, `shared/`)

---

## Key File Locations

| What | Where |
|------|-------|
| Tutorial CLI | `src/main.py` |
| Re-clip tool | `src/reclip.py` |
| Frame interpolation | `src/interpolate.py` |
| RIFE on Modal | `src/rife_modal.py` |
| Gemini analysis | `src/video_analyzer.py` |
| API key env var | `GOOGLE_API_KEY` |
| Output template | `templates/tutorial.md` |
| Generated tutorials | `output/` folder |
| Session logs | `AGENTS/session-log-*.md` |
| Interpolation research | `AGENTS/frame-interpolation-options.md` |
