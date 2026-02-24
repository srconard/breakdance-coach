# Session Log - January 22-23, 2026

## Project: Breakdance Tutorial GIF Generator

### Session Overview
Built a Python tool to convert YouTube breakdancing tutorials into step-by-step GIFs with AI-generated descriptions, outputting Obsidian-compatible Markdown files.

---

## What Was Built

### Core Architecture
```
YouTube URL (or local file) → Download → Preprocess → Gemini Analysis → LLM Descriptions → GIFs → Markdown
```

### Files Created

| File | Purpose |
|------|---------|
| `src/main.py` | CLI entry point, orchestrates entire pipeline |
| `src/downloader.py` | YouTube video downloading via yt-dlp |
| `src/video_prep.py` | Video preprocessing (downscale, fps reduction, trimming) |
| `src/video_analyzer.py` | Gemini API integration for step identification |
| `src/description.py` | Multi-provider LLM descriptions (Google/Anthropic/OpenAI) |
| `src/gif_creator.py` | FFmpeg-based GIF generation with palette optimization |
| `src/output.py` | Obsidian Markdown generation |
| `config.py` | Settings and API key management |
| `templates/tutorial.md` | Jinja2 template for output |
| `requirements.txt` | Python dependencies |
| `CLAUDE.md` | Project context for Claude Code |
| `README.md` | User documentation |
| `AGENTS/*.md` | AI agent documentation |

### Features Implemented
- ✅ YouTube video downloading (blocked by throttling)
- ✅ **Local file support** (`--local-file` flag)
- ✅ Video preprocessing with cost reduction options
- ✅ Gemini video analysis for step identification
- ✅ Multi-provider LLM support (Google, Anthropic, OpenAI)
- ✅ Optimized GIF creation with palette generation
- ✅ Obsidian-compatible Markdown output
- ✅ CLI with configurable options
- ✅ Trim intro/outro settings
- ✅ Deno JavaScript runtime installed and configured

---

## Issues Encountered & Resolution Attempts

### 1. yt-dlp YouTube Download Failures
**Problem:** YouTube has recently tightened restrictions.

**Errors Seen:**
```
HTTP Error 403: Forbidden
No supported JavaScript runtime could be found
0 bytes read, X more expected (throttling)
```

**Resolution Attempts:**
1. ❌ Different format selections (`best[ext=mp4]`, format 18)
2. ❌ Different player clients (android, ios, tv, web_creator, android_vr)
3. ✅ Installed Deno JavaScript runtime (fixed JS runtime warning)
4. ❌ Added Deno to PATH in downloader.py
5. ❌ Increased retries and reduced chunk size
6. ❌ Added sleep intervals between requests

**Current Status:** Deno is working (no more JS runtime warnings), but YouTube is actively throttling/blocking downloads. Connection drops after ~290KB downloaded.

**Workaround Implemented:** Added `--local-file` flag to use pre-downloaded videos.

### 2. Cookie Extraction Issues
- Chrome cookies couldn't be accessed (browser was open/locked)
- Permission denied errors when trying to copy cookie database

### 3. Deprecated Google API
- `google.generativeai` package is deprecated
- Should migrate to `google.genai` package (not yet done)

---

## Current State (End of Session)

### What Works
- ✅ Deno installed at: `C:\Users\Shawn\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe\deno.exe`
- ✅ Deno added to PATH in `src/downloader.py`
- ✅ Local file support via `--local-file` flag
- ✅ All Python dependencies installed
- ✅ FFmpeg 6.0 installed and working

### What's Blocked
- ❌ YouTube downloads fail due to aggressive throttling
- ⚠️ Full pipeline untested (waiting for working video input)

### Ready to Test
Once a local video file is provided:
```bash
python -m src.main --local-file "video.mp4" --title "Tutorial Name"
```

---

## Configuration Notes
- Google API key added directly to `config.py` line 44
- Deno path hardcoded in `src/downloader.py` line 12-14

---

## Test Video Attempted
- URL: https://www.youtube.com/watch?v=oOcy7xVRwyQ
- Title: "BEST FLARE TUTORIAL (2020) - BY SAMBO - HOW TO BREAKDANCE (#5)"
- Result: Download failed due to YouTube throttling (not auth issues)

---

## Dependencies Installed
All packages installed successfully:
- yt-dlp (2025.12.8)
- google-generativeai (0.8.6)
- anthropic (0.76.0)
- openai (2.15.0)
- ffmpeg-python (0.2.0)
- jinja2 (3.1.6)
- **Deno 2.6.5** (system-level, via winget)

System requirements confirmed:
- ffmpeg 6.0 ✅
- Deno 2.6.5 ✅
- Python 3.10.7 ✅
