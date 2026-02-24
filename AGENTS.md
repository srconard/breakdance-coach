# AGENTS.md - Project Context for AI Agents

## What Is This Project?

**Breakdance Coach** — A suite of AI-powered tools for learning breakdancing:

1. **Tutorial Wiki Generator** — Converts YouTube videos into step-by-step visual tutorials (Obsidian markdown with embedded GIF/MP4)
2. **3D Move Analyzer** (planned) — Generates interactive 3D models from breakdancing videos (GVHMR → SMPL → GLB → Obsidian)
3. **Shared Tools** — Frame interpolation, video download, cloud GPU utilities

## Documentation Locations

| File | What It Contains |
|------|------------------|
| `AGENTS/PROJECT-ROADMAP.md` | **Master roadmap** — status, 3D feature spec, reorg plan, all future work |
| `AGENTS/frame-interpolation-options.md` | Research: RIFE, fal.ai, Replicate, Topaz comparison |
| `AGENTS/CONTEXT.md` | Technical architecture, code patterns, API usage |
| `AGENTS/session-log-2026-02-23.md` | Latest: YouTube fix, GitHub push, interpolation |
| `AGENTS/session-log-2026-01-23.md` | Pipeline testing & fixes |
| `AGENTS/session-log-2026-01-22.md` | Initial build session |
| `README.md` | User-facing documentation |

## Key Files (Tutorial Generator)

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point - orchestrates the pipeline |
| `src/video_analyzer.py` | Gemini integration (`gemini-2.5-flash`) |
| `src/description.py` | Multi-provider LLM descriptions (`gemini-2.5-flash`) |
| `src/downloader.py` | YouTube downloads (android_vr/tv client workaround) |
| `src/interpolate.py` | Frame interpolation (FFmpeg, upgrades planned) |
| `config.py` | Settings and API key management via env vars |

## Current Status

🟢 **Tutorial generator working.** 3D analyzer planned.

- GitHub: https://github.com/srconard/breakdance-coach
- Gemini model: `gemini-2.5-flash`
- API key: `GOOGLE_API_KEY` environment variable

## Common Commands

```bash
# Generate tutorial
python -m src.main --local-file "video.mp4" --title "Tutorial" --format mp4

# Generate tutorial from YouTube
python -m src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# Slow-mo a clip
python -m src.interpolate "output/tutorial/gifs/step_05.mp4" --slowdown 3
```

## When Resuming Work

1. Check `AGENTS/PROJECT-ROADMAP.md` for full roadmap and status
2. Open tasks: Refine Gemini prompt, upgrade interpolation, 3D analyzer
3. Frame interpolation research: `AGENTS/frame-interpolation-options.md`
4. 3D feature spec: in PROJECT-ROADMAP.md under "Feature Spec: 3D Move Analyzer"
