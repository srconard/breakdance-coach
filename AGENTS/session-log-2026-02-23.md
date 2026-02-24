# Session Log - February 23, 2026

## Project: Breakdance Coach

### Session Overview
Generated a new tutorial from YouTube, fixed model issues, pushed to GitHub, built frame interpolation tool, researched advanced interpolation options, designed 3D move analyzer feature, and planned project reorganization.

---

## What Was Done

### 1. YouTube Download Success
- **Video:** "FLARE Workout Exercises (Intermediate/Advanced)" (9.77MB)
- **URL:** https://www.youtube.com/watch?v=Ja9JeWhpHms
- **Result:** Downloaded successfully using yt-dlp with android_vr/tv client workaround
- YouTube download issue (documented as broken) actually works for some videos

### 2. Fixed Retired Gemini Models
- `gemini-exp-1206` → `gemini-2.5-flash` (video_analyzer.py)
- `gemini-exp-1206` → `gemini-2.5-flash` (description.py)
- Listed available models via API to find working alternatives

### 3. Generated FLARE Workout Tutorial
- **18 exercises identified** by Gemini (all with visual demonstrations)
- **18 MP4 clips** created from original video
- **18 AI-generated descriptions** with body positioning cues
- Output: `output/FLARE_Workout_Exercises/`

### 4. Published to GitHub
- Scrubbed API keys from CLAUDE.md, PROJECT-ROADMAP.md, session logs
- Created .gitignore (excludes .claude/, .obsidian/, downloads/, output/)
- Reinitialized git with clean history (no API keys in any commits)
- Created public repo: https://github.com/srconard/breakdance-coach

### 5. Built Frame Interpolation Tool
- Created `src/interpolate.py` - standalone slow-mo video generator
- Uses FFmpeg minterpolate (optical flow based)
- Configurable slowdown factor and output FPS
- Tested on L-Sit to V-Sit Circles (3x) and Tuck Planche (4x) — both worked
- Quality: acceptable but not great on fast complex motion (ghosting/smearing)

### 6. Frame Interpolation Research
- Researched all major interpolation tools (local and cloud)
- Wrote comprehensive options doc: `AGENTS/frame-interpolation-options.md`
- **Best options identified:**
  - RIFE (local or Modal.com) — best quality, free, needs GPU
  - fal.ai / Replicate APIs — easiest cloud integration, good quality
  - Topaz Apollo — highest quality (dance-specific), expensive
- FFmpeg minterpolate stays as zero-dependency fallback

### 7. 3D Move Analyzer Feature Design
- Researched 3D pose estimation from video (GVHMR, PromptHMR, WHAM, 4D-Humans)
- Researched SMPL body model and mesh generation pipeline
- Researched viewing options (Obsidian model-viewer plugin, Three.js, `<model-viewer>`)
- Researched cloud GPU options (Meshcapade, DeepMotion, Modal.com)
- Wrote full feature spec in PROJECT-ROADMAP.md

### 8. Project Reorganization Plan
- Designed monorepo structure: `tutorial-generator/`, `3d-analyzer/`, `shared/`
- Rationale: separate tools with different dependencies, shared utilities
- Will execute reorg when starting 3D analyzer

### 9. Documentation Updates
- Rewrote all docs to reflect "Breakdance Coach" as a multi-tool project
- Updated CLAUDE.md, AGENTS.md, PROJECT-ROADMAP.md, CONTEXT.md
- All docs now reference 3D analyzer as planned feature

---

## Code Changes

### Files Modified
1. `src/video_analyzer.py` — Model: `gemini-exp-1206` → `gemini-2.5-flash`
2. `src/description.py` — Model: `gemini-exp-1206` → `gemini-2.5-flash`
3. `CLAUDE.md` — Full rewrite: multi-tool vision, 3D pipeline diagram
4. `AGENTS.md` — Full rewrite: multi-tool vision, doc index
5. `AGENTS/PROJECT-ROADMAP.md` — Full rewrite: reorg plan, 3D feature spec, interpolation status
6. `AGENTS/CONTEXT.md` — Updated architecture, tested status

### Files Created
1. `.gitignore` — Excludes .claude/, .obsidian/, downloads/, output/
2. `src/interpolate.py` — Frame interpolation tool (FFmpeg minterpolate)
3. `AGENTS/session-log-2026-02-23.md` — This file
4. `AGENTS/frame-interpolation-options.md` — Interpolation research doc

---

## Tutorial Output: FLARE Workout Exercises

18 steps identified:
1. Seated Pike Lifts
2. Seated Straddle Lifts (1 Leg)
3. Seated Straddle Lifts
4. Seated Straddle Extensions
5. L-Sit Open & Close
6. L-Sit Wide Legs
7. Half Straddle
8. Straddle
9. V-Sit
10. L-Sit to Tuck
11. L-Sit to V-Sit Circles
12. Bodyweight Hip Thrust
13. Flare Side Hold
14. Tuck L-Sit
15. Exercise for the Tuck L-Sit
16. Tuck Planche
17. Exercise for the Tuck Planche
18. L-Sit to Tuck Planche

---

## Research Findings Summary

### Frame Interpolation
- RIFE v4.25/v4.26 is the community standard (MIT, fast, GPU)
- SG-RIFE (Dec 2025 paper) adds semantic guidance — watch for release
- fal.ai hosts FILM video endpoint with scene detection
- Replicate hosts ST-MFNet (up to 16x) and FILM
- Topaz Apollo is best quality but $50+/mo
- Tip: 2x iterative is better than single 4x for fast motion

### 3D Pose Estimation
- GVHMR (SIGGRAPH Asia 2024) — gravity-aware, best for inversions/freezes
- PromptHMR (CVPR 2025) — newest SOTA, accepts language prompts
- SMPL body model: 6,890 vertices, 72 pose params, 10 shape params
- Pipeline: Video → YOLO+ViTPose → GVHMR → smplx → Blender → .GLB
- Obsidian plugin exists for 3D model viewing: `![[move.glb#autoplay]]`
- Commercial APIs: Meshcapade (10-99 EUR/mo), DeepMotion (freemium)
