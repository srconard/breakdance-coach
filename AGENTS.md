# CLAUDE.md - Project Context for Claude Code

## What Is This Project?

**Breakdance Tutorial GIF Generator** - A Python CLI tool that converts YouTube breakdancing tutorial videos into step-by-step GIFs with AI-generated descriptions, outputting Obsidian-compatible Markdown files.

The goal is to help learn breakdancing by creating visual reference material from video tutorials.

## Project Architecture

```
YouTube URL
    ↓
[yt-dlp] Download video with audio
    ↓
[ffmpeg] Preprocess (downscale, reduce fps, trim)
    ↓
[Gemini API] Analyze video → identify steps with timestamps
    ↓
[Configurable LLM] Generate descriptions (Google/Anthropic/OpenAI)
    ↓
[ffmpeg] Create optimized GIFs for each step
    ↓
[Jinja2] Generate Obsidian Markdown with embedded GIFs
```

## Documentation Locations

All detailed documentation is in the `AGENTS/` folder:

| File | What It Contains |
|------|------------------|
| `AGENTS/CONTEXT.md` | Technical architecture, code patterns, API usage |
| `AGENTS/PROJECT-ROADMAP.md` | Current status, blockers, next steps, future plans |
| `AGENTS/session-log-2026-01-22.md` | Build history and decisions made |
| `README.md` | User-facing documentation and usage instructions |

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point - orchestrates the entire pipeline |
| `src/video_analyzer.py` | Gemini integration - contains `TutorialStep` dataclass |
| `src/description.py` | Multi-provider LLM support (Google, Anthropic, OpenAI) |
| `config.py` | Settings and API keys - **user added Google API key on line 44** |

## Current Status

🟡 **Core code complete, blocked on YouTube downloads**

YouTube now requires:
- Deno JavaScript runtime (`winget install DenoLand.Deno`)
- PO Tokens for some videos
- Browser cookies for authenticated content

**To test the pipeline:** Either fix downloads or add `--local-file` flag support.

## Important Context

1. **API Key Location**: User added their Google API key directly in `config.py` line 44 (not via environment variable)

2. **Deprecated Package**: `google.generativeai` is deprecated - should migrate to `google.genai`

3. **Output Format**: Obsidian Markdown with wiki-style embeds: `![[gifs/step_01.gif]]`

4. **Cost Optimization**: Video preprocessing (480p, 15fps) reduces Gemini API costs significantly

## Common Commands

```bash
# Run the tool
python -m src.main "https://youtube.com/watch?v=..."

# With options
python -m src.main "URL" --trim-intro 10 --description-model anthropic

# Test individual modules
python -m src.video_prep <video.mp4>
python -m src.gif_creator <video.mp4> <start_sec> <end_sec>
```

## When Resuming Work

1. Check `AGENTS/PROJECT-ROADMAP.md` for current blockers and next steps
2. The YouTube download issue is the priority - see roadmap for solutions
3. All code is written but untested end-to-end due to download blocker
