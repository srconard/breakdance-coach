# AGENTS.md - Project Context for AI Agents

Last updated: 2026-08-04

## What This Project Is

Breakdance Coach is a monorepo with four tools:

1. Tutorial Wiki Generator (`tutorial_generator/`): converts videos into step-by-step Obsidian tutorials with embedded GIF/MP4/WebM clips.
2. 3D Move Analyzer (`analyzer_3d/`): generates animated 3D models from dance videos (GVHMR -> SMPL -> GLB -> Obsidian).
3. Shared Utilities (`shared/`): downloader, frame interpolation, cloud GPU helpers.
4. Dance Mirror (`dance_mirror/`): turns short social or local dance clips into self-contained practice lessons.

## Current Status

- Tutorial generator: working end-to-end.
- Re-clip flow: working (metadata-driven HQ re-extraction).
- Frame interpolation: FFmpeg local fallback + RIFE on Modal.
- 3D analyzer: pipeline implemented end-to-end; Obsidian playback/embedding UX still being refined.
- Dance Mirror: first offline-capable video lesson slice implemented; Gemini and URL paths await connected smoke tests.

GitHub: https://github.com/srconard/breakdance-coach

## Documentation Map

| File | Purpose |
|------|---------|
| `AGENTS/PROJECT-ROADMAP.md` | Master roadmap, active priorities, 3D spec |
| `AGENTS/CONTEXT.md` | Technical architecture and implementation notes |
| `AGENTS/frame-interpolation-options.md` | Interpolation research and backend comparison |
| `AGENTS/session-log-2026-02-24.md` | Latest detailed session log |
| `AGENTS/session-log-2026-08-04-dance-mirror.md` | Dance Mirror slice 1 implementation and validation |
| `AGENTS/session-log-2026-02-23.md` | YouTube fix and interpolation history |
| `AGENTS/session-log-2026-01-23.md` | Pipeline test and fix history |
| `AGENTS/session-log-2026-01-22.md` | Initial implementation log |
| `README.md` | User-facing quick start |

## Environment Setup

### Required

- Python 3.10+
- FFmpeg available on PATH
- Google Gemini API key (`GOOGLE_API_KEY`)

### Optional

- Deno for yt-dlp JavaScript runtime fallback
- Modal account and auth for cloud GPU tasks (RIFE and GVHMR)
- Anthropic/OpenAI keys for non-Google description provider mode

### Install

```bash
pip install -r requirements.txt
```

### Environment Variables

Required:

```bash
# Windows
set GOOGLE_API_KEY=your_google_api_key

# macOS/Linux
export GOOGLE_API_KEY=your_google_api_key
```

Optional:

```bash
set ANTHROPIC_API_KEY=your_anthropic_key
set OPENAI_API_KEY=your_openai_key
```

Modal auth (one-time, optional):

```bash
python -m modal setup
```

## Directory Map

| Path | Purpose | Safe to delete/regenerate |
|------|---------|---------------------------|
| `tutorial_generator/src/` | Tutorial pipeline code | No |
| `tutorial_generator/templates/` | Markdown template(s) | No |
| `tutorial_generator/config.py` | Config and provider settings | No |
| `analyzer_3d/src/` | 3D analysis and export pipeline | No |
| `analyzer_3d/templates/` | 3D markdown template(s) | No |
| `dance_mirror/src/` | Dance lesson CLI, segmentation, schema, and rendering | No |
| `dance_mirror/templates/` | Self-contained practice page template | No |
| `shared/` | Shared downloader/interpolation utilities | No |
| `output/` (runtime) | Generated tutorials and media artifacts | Yes |
| temp/intermediate files (runtime) | Preprocessed/transient media | Yes |

## Key Runtime Commands

### Tutorial Generator

```bash
# Local file input
python -m tutorial_generator.src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# YouTube input
python -m tutorial_generator.src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# Re-clip single step at high quality from metadata
python -m tutorial_generator.src.reclip "output/My_Tutorial" --step 11 --download-hq 1080p
```

### Frame Interpolation

```bash
# Local FFmpeg backend
python -m shared.interpolate "output/My_Tutorial/gifs/step_05.mp4" --slowdown 3 --backend ffmpeg

# Cloud RIFE backend (Modal)
python -m shared.interpolate "output/My_Tutorial/gifs/step_05.mp4" --slowdown 3 --backend rife
```

### 3D Analyzer

```bash
# One-time GVHMR setup
python -m analyzer_3d.src.gvhmr_setup

# Deploy GVHMR app to Modal
python -m modal deploy analyzer_3d/src/gvhmr_modal.py

# Analyze and export 3D result
python -m analyzer_3d.src.main "video.mp4" --backend gvhmr
```

## Validation and Quality Gates

There is currently no formal test suite in this repository. Use these checks before considering work complete:

1. Tutorial command runs without exceptions.
2. Output folder contains markdown, media clips, and `tutorial_metadata.json`.
3. At least one extracted step has valid timestamps and playable media.
4. Markdown references generated media files correctly.
5. If 3D path is touched: GLB exports and can be loaded by target viewer.

## Definition of Done

### Tutorial Generator Changes

- Pipeline completes for at least one known input (local video preferred).
- Step segmentation output is coherent (no empty/overlapping critical steps).
- Generated markdown renders in Obsidian with working embedded media.
- Metadata JSON is present and usable by `reclip`.

### 3D Analyzer Changes

- GVHMR stage produces valid SMPL params/mesh payload.
- Blender export stage produces a `.glb` with armature + animation.
- Coordinate transform remains correct (`(x, y, z) -> (x, -z, y)`).
- Obsidian markdown references the generated 3D artifact.

## Failure Handling Playbook

### YouTube download failures (403/auth/runtime)

- Confirm yt-dlp path and JS runtime availability.
- Install/verify Deno.
- Retry with alternate client mode if supported by downloader code.
- Fall back to `--local-file` flow if needed.

### Gemini/API failures

- Verify `GOOGLE_API_KEY` is set in current shell.
- Check quota/rate limits and retry with backoff.
- Reduce clip length/input size if request payload is too large.

### FFmpeg errors

- Confirm FFmpeg is installed and available on PATH.
- Validate input path and codec support.
- Retry with simpler output settings (e.g., lower FPS/width) for diagnosis.

### Modal errors (RIFE/GVHMR)

- Re-run `python -m modal setup` for auth.
- Verify deployment target/function name.
- Confirm required model/checkpoint assets are present in Modal volumes.

## Model and Provider Policy

- Video analysis model: `gemini-2.5-flash` (default and preferred).
- Description generation provider order: Google default, Anthropic/OpenAI optional.
- Prioritize reliability and cost control over highest-quality model choice.
- Do not switch provider/model defaults without documenting reason in session log.

## 3D Analyzer Milestones and Exit Criteria

1. Inference stability: GVHMR run succeeds on representative clips.
2. Mesh extraction integrity: SMPL mesh/topology + weights are valid.
3. Export integrity: Blender headless export emits playable GLB.
4. Viewer integration: Obsidian playback flow documented and repeatable.
5. UX hardening: clear CLI usage, errors, and artifact paths.

Each milestone is complete only when artifacts are generated and validated locally.

## Data, Security, and Git Hygiene

Never commit:

- `.env` files or raw API keys
- Modal secrets or private tokens
- Large generated media/checkpoints unless intentionally versioned

Recommended ignores (verify in `.gitignore`):

- `output/`
- cache/temp directories
- local model/checkpoint downloads

## Change Workflow

- Branch naming: `codex/<short-topic>`.
- Keep commits focused and descriptive.
- Prefer non-destructive operations; do not reset unrelated work.
- If unexpected unrelated file changes appear, stop and confirm before proceeding.

## Key Files

| File | Purpose |
|------|---------|
| `tutorial_generator/src/main.py` | Tutorial CLI entrypoint |
| `tutorial_generator/src/video_analyzer.py` | Gemini step detection |
| `tutorial_generator/src/description.py` | Description generation across providers |
| `tutorial_generator/src/gif_creator.py` | Clip generation (gif/mp4/webm) |
| `tutorial_generator/src/output.py` | Markdown + metadata output |
| `tutorial_generator/src/reclip.py` | Metadata-driven HQ clip re-extraction |
| `tutorial_generator/config.py` | Config and API key access |
| `shared/downloader.py` | YouTube downloading |
| `shared/interpolate.py` | Interpolation backend selection |
| `shared/rife_modal.py` | RIFE cloud backend on Modal |
| `analyzer_3d/src/main.py` | 3D analyzer CLI entrypoint |
| `analyzer_3d/src/gvhmr_modal.py` | GVHMR Modal integration |
| `analyzer_3d/src/gvhmr_setup.py` | Checkpoint setup and upload |
| `analyzer_3d/src/exporter.py` | SMPL to GLB export path |
| `analyzer_3d/src/blender_scripts/smpl_to_glb.py` | Blender conversion script |
| `analyzer_3d/src/output.py` | 3D markdown output generation |
| `analyzer_3d/src/phase_trace.py` | CPU phase trace over cleaned SMPL pkls (heights, rotation cycles, inferred support) -> `<stem>_phases.json` |
| `dance_mirror/src/main.py` | Dance Mirror build/refresh CLI entrypoint |
| `dance_mirror/src/segmenter.py` | Gemini part segmentation and uniform offline fallback |
| `dance_mirror/src/lesson.py` | Editable `dance_lesson.json` validation and persistence |
| `dance_mirror/src/render.py` | Self-contained lesson HTML renderer |
| `dance_mirror/templates/lesson.html` | Mobile-first offline practice UI |

## Quick Resume Checklist

1. Read `AGENTS/PROJECT-ROADMAP.md` for current priorities.
2. Read newest `AGENTS/session-log-*.md` entry.
3. Confirm environment (`GOOGLE_API_KEY`, FFmpeg, optional Modal auth).
4. Run one tutorial generation smoke command.
5. If working on 3D pipeline, run one known-good analyzer command and verify GLB output.

## Recent Decisions

- Monorepo layout (`tutorial_generator/`, `analyzer_3d/`, `shared/`) is the source of truth.
- Keep Gemini Flash as default analysis model.
- Keep FFmpeg as local interpolation fallback; RIFE via Modal for quality mode.
- Preserve metadata-first workflow to enable re-clip without re-running AI analysis.
