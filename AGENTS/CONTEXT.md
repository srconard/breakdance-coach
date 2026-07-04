# AI Agent Context - Breakdance Coach

Last updated: 2026-03-06

## Quick Status

- Tutorial pipeline works end-to-end.
- Re-clip workflow works from saved metadata.
- Interpolation has two backends: FFmpeg local and RIFE on Modal.
- 3D pipeline is implemented; viewer integration/UX is the main open area.

## Primary Commands

```bash
# Tutorial from local file
python -m tutorial_generator.src.main --local-file "video.mp4" --title "Tutorial" --format mp4

# Tutorial from YouTube
python -m tutorial_generator.src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# Re-clip using metadata
python -m tutorial_generator.src.reclip "output/My_Tutorial" --step 3 --download-hq 1080p

# Slow-mo clip (local)
python -m shared.interpolate "clip.mp4" --slowdown 3 --backend ffmpeg

# Slow-mo clip (cloud)
python -m shared.interpolate "clip.mp4" --slowdown 3 --backend rife

# 3D analysis
python -m analyzer_3d.src.main "video.mp4" --backend gvhmr
```

## Architecture

```
Video input (local or YouTube)
  -> downloader/preprocess
  -> Gemini step extraction
  -> description generation
  -> clip rendering from original video
  -> markdown + metadata output
```

3D path:

```
Video input
  -> GVHMR inference (Modal)
  -> SMPL mesh/pose data
  -> Blender headless GLB export
  -> markdown embed output
```

## Key Design Decisions

- Gemini Flash is the default for step extraction.
- Keep tutorial media sourced from the original video quality.
- Persist metadata to enable re-clipping without re-running AI analysis.
- Keep FFmpeg backend as local fallback for interpolation.

## Critical File Locations

| What | Where |
|------|-------|
| Tutorial orchestrator | `tutorial_generator/src/main.py` |
| Step extraction prompt and parsing | `tutorial_generator/src/video_analyzer.py` |
| Description generation | `tutorial_generator/src/description.py` |
| Clip generation | `tutorial_generator/src/gif_creator.py` |
| Markdown + metadata output | `tutorial_generator/src/output.py` |
| Re-clip command | `tutorial_generator/src/reclip.py` |
| Downloader | `shared/downloader.py` |
| Interpolation | `shared/interpolate.py` |
| RIFE Modal integration | `shared/rife_modal.py` |
| 3D CLI | `analyzer_3d/src/main.py` |
| GVHMR Modal flow | `analyzer_3d/src/gvhmr_modal.py` |
| 3D export | `analyzer_3d/src/exporter.py` |

## Current Risks

1. Step extraction can include talking-head sections if prompt is too broad.
2. Gemini/download API failures need stronger retry and diagnostics.
3. Gemini SDK migration to `google.genai` is pending.
4. Test coverage is minimal.

## Minimal Validation Checklist

1. Run tutorial CLI on a known local test clip.
2. Confirm markdown, media clips, and `tutorial_metadata.json` exist.
3. Verify at least one re-clip command succeeds from metadata.
4. If 3D-related code changes, verify `.glb` output loads and animates.
