# Breakdance Coach - Project Roadmap

Last updated: 2026-03-06

## Vision

Breakdance Coach is a suite of AI-powered tools for learning breakdancing:

1. Tutorial Wiki Generator: Video -> step-by-step Obsidian tutorial with media clips.
2. 3D Move Analyzer: Video -> animated 3D model (GVHMR -> SMPL -> GLB).
3. Shared Tools: Download, interpolation, and cloud GPU utilities used by both tools.

## Repository Structure (Current)

```
breakdance-coach/
|-- tutorial_generator/
|   |-- src/
|   |-- templates/
|   `-- config.py
|-- analyzer_3d/
|   |-- src/
|   `-- templates/
|-- shared/
|-- AGENTS/
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
`-- requirements.txt
```

## Current Status

### Tutorial Generator

- Working end-to-end for local files and YouTube URLs.
- Uses `gemini-2.5-flash` for video step extraction.
- Outputs markdown + media + `tutorial_metadata.json`.
- Re-clip tool supports high-quality re-extraction from metadata.

### Shared Interpolation

- FFmpeg backend available locally.
- RIFE backend available via Modal GPU.

### 3D Analyzer

- End-to-end pipeline implemented: GVHMR inference -> SMPL mesh data -> Blender GLB export -> markdown output.
- Remaining work is mainly viewer UX hardening and documentation of the final Obsidian workflow.

GitHub: https://github.com/srconard/breakdance-coach

## Active Priorities

1. Refine Gemini step-detection prompt to exclude non-demonstration talk segments.
2. Improve reliability: retries/backoff and clearer error handling around API/download steps.
3. Migrate from `google.generativeai` to `google.genai`.
4. Add minimal smoke tests and CI checks.
5. Finalize and document the 3D viewer path in Obsidian.

## Open Tasks

### 1) Prompt Quality: Demonstration-Only Steps

- File: `tutorial_generator/src/video_analyzer.py`
- Goal: Extract only segments with physically demonstrated movement or position.
- Acceptance: fewer talking-head segments in output on known sample videos.

### 2) Reliability Hardening

- Add retry policy (with backoff) for Gemini calls and downloads.
- Improve diagnostics for quota, auth, and ffmpeg failure cases.

### 3) SDK Migration

- Migrate Gemini integration from `google.generativeai` to `google.genai`.
- Re-verify end-to-end tutorial generation after migration.

### 4) Automated Checks

- Add smoke tests around CLI output structure and metadata generation.
- Add lightweight CI workflow for lint/smoke validation.

### 5) 3D Viewer Completion

- Standardize Obsidian embedding workflow and validate animation playback.
- Document exact plugin/settings requirements.

## Completed Milestones

- RIFE interpolation deployed on Modal with working slow-motion output behavior.
- Re-clip workflow and metadata JSON export implemented.
- Multi-quality YouTube download support added.
- Monorepo structure implemented (`tutorial_generator/`, `analyzer_3d/`, `shared/`).

## Technical Debt (Current)

1. Migrate from `google.generativeai` to `google.genai`.
2. Add proper retry logic with exponential backoff.
3. Add unit/smoke tests and CI.
4. Ensure downloader runtime discovery does not depend on hardcoded paths.
5. Improve URL/input validation and user-facing error messages.

## Key Paths

| What | Where |
|------|-------|
| Tutorial CLI | `tutorial_generator/src/main.py` |
| Video step analyzer | `tutorial_generator/src/video_analyzer.py` |
| Re-clip tool | `tutorial_generator/src/reclip.py` |
| Interpolation CLI | `shared/interpolate.py` |
| RIFE Modal backend | `shared/rife_modal.py` |
| 3D analyzer CLI | `analyzer_3d/src/main.py` |
| GVHMR Modal backend | `analyzer_3d/src/gvhmr_modal.py` |
| GLB exporter pipeline | `analyzer_3d/src/exporter.py` |
| Docs and logs | `AGENTS/` |
