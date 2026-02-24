# CLAUDE.md - Project Context for Claude Code

**Last Updated:** January 24, 2026

## What Is This Project?

**Breakdance Tutorial GIF Generator** - A Python CLI tool that converts YouTube breakdancing tutorial videos into step-by-step GIFs/videos with AI-generated descriptions, outputting Obsidian-compatible Markdown files.

## Current Status: 🟡 Working - Refinements Needed

Pipeline works end-to-end. Two open tasks remain before production-ready.

### What's Working
- ✅ Video preprocessing (71% size reduction)
- ✅ Gemini video analysis (step identification)
- ✅ Description generation (with rate limiting)
- ✅ GIF/MP4/WebM output (from original video)
- ✅ Obsidian-compatible markdown

### Open Tasks

#### 1. 🔴 Refine Gemini Prompt
**File:** `src/video_analyzer.py` (lines 82-102)

**Problem:** Currently identifies ALL sections including talking-head segments without visual demonstrations.

**Goal:**
- Only extract steps with meaningful visual components (actual move demonstrations)
- Skip sections where instructor is just talking
- Output fuller, more detailed tutorial descriptions

#### 2. 🔴 Fix YouTube Downloads
**File:** `src/downloader.py`

**Problem:** YouTube throttling blocks yt-dlp downloads after ~290KB.

**Possible fixes:**
- Browser cookies authentication
- Alternative download libraries
- PO token generation

---

## Quick Start

```bash
# Set API key
set GOOGLE_API_KEY=your_google_api_key_here

# Run with MP4 output (recommended - small files, playback controls)
python -m src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# Run with GIF output
python -m src.main --local-file "video.mp4" --title "Tutorial Name" --format gif
```

## Project Architecture

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

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point |
| `src/video_analyzer.py` | **Gemini prompt here (lines 82-102)** - needs refinement |
| `src/downloader.py` | **YouTube download** - needs fixing |
| `src/description.py` | Multi-provider LLM with rate limiting |
| `src/gif_creator.py` | GIF/MP4/WebM creation |
| `src/output.py` | Obsidian markdown generation |
| `config.py` | API key via `GOOGLE_API_KEY` env var |

## Output Formats

| Format | Size | Playback Controls | Command |
|--------|------|-------------------|---------|
| MP4 | ~29 MB | Full (scrub, pause) | `--format mp4` |
| WebM | Smallest | Full | `--format webm` |
| GIF | ~236 MB | None (auto-loop) | `--format gif` |

## Common Commands

```bash
# MP4 output (recommended)
python -m src.main --local-file "video.mp4" --title "Tutorial" --format mp4

# GIF output
python -m src.main --local-file "video.mp4" --title "Tutorial" --format gif

# With trimming
python -m src.main --local-file "video.mp4" --title "Tutorial" --format mp4 --trim-intro 10

# Custom quality (smaller files)
python -m src.main --local-file "video.mp4" --title "Tutorial" --format mp4 --fps 10 --width 480

# Use Anthropic for descriptions
python -m src.main --local-file "video.mp4" --title "Tutorial" --description-model anthropic
```

## Documentation

| File | Contents |
|------|----------|
| `AGENTS/PROJECT-ROADMAP.md` | Full status, open tasks, blockers |
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
