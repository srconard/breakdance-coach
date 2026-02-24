# AI Agent Context - Breakdance Coach Project

**Last Updated:** January 23, 2026

## Quick Status

🟡 **Ready for testing with local files.** YouTube downloads blocked by throttling.

```bash
# To test the full pipeline:
python -m src.main --local-file "video.mp4" --title "Tutorial Name"
```

## Project Purpose

A tool to help learn breakdancing by converting YouTube tutorial videos into digestible, step-by-step GIF tutorials with descriptions. Output is designed for Obsidian note-taking.

## Architecture

```
Video Input (local file or YouTube URL)
    ↓
[ffmpeg] Preprocess (480p, 15fps, trim)
    ↓
[Gemini API] Analyze video → steps + timestamps
    ↓
[LLM] Generate descriptions (Google/Anthropic/OpenAI)
    ↓
[ffmpeg] Create GIFs (two-pass palette encoding)
    ↓
[Jinja2] Obsidian Markdown output
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Gemini for video analysis | Native video upload, cost effective |
| Multi-provider descriptions | User wants to compare LLMs |
| Obsidian Markdown output | User's note-taking system |
| GIFs (not video clips) | Auto-loop, universal playback |
| Cost reduction defaults | 480p + 15fps reduces API costs |
| Two-pass GIF encoding | Better quality, smaller files |

## Critical File Locations

| What | Where | Notes |
|------|-------|-------|
| CLI entry | `src/main.py` | Orchestrates pipeline |
| Deno PATH | `src/downloader.py:12-14` | Hardcoded path |
| Google API key | `config.py:44` | Directly in file |
| Core dataclass | `src/video_analyzer.py` | `TutorialStep` |
| Output template | `templates/tutorial.md` | Jinja2 |

## Current Blockers

### YouTube Downloads (❌ Blocked)
- **Symptom:** Downloads fail after ~290KB with "0 bytes read" errors
- **Cause:** YouTube throttling (not auth, not missing runtime)
- **Deno status:** ✅ Installed and working (no more JS warnings)
- **Workaround:** `--local-file` flag

### Deprecated API (⚠️ Warning Only)
- `google.generativeai` is deprecated
- Still functional, just shows warning
- Should migrate to `google.genai` eventually

## Code Patterns

### TutorialStep Dataclass
```python
@dataclass
class TutorialStep:
    step_number: int
    start_time: str      # "MM:SS"
    end_time: str        # "MM:SS"
    label: str           # Short description
    start_seconds: float # Auto-computed
    end_seconds: float   # Auto-computed
```

### Gemini Video Upload Pattern
```python
video_file = genai.upload_file(path=str(video_path))
while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = genai.get_file(video_file.name)
```

### GIF Creation (Two-Pass)
1. Generate color palette from video segment
2. Create GIF using palette for better colors/smaller size

## What's Been Tested

| Component | Tested | Notes |
|-----------|--------|-------|
| Dependencies install | ✅ | All packages working |
| Deno runtime | ✅ | No more JS warnings |
| YouTube download | ❌ | Blocked by throttling |
| Full pipeline | ❌ | Needs local video file |

## For Next Session

1. **Get a local video file** to test full pipeline
2. **Run:** `python -m src.main --local-file "video.mp4" --title "Test"`
3. **Verify:** Preprocessing → Gemini → Descriptions → GIFs → Markdown
4. **Then:** Fix any issues found in end-to-end test

## Dependencies

All installed and working:
- yt-dlp 2025.12.8
- google-generativeai 0.8.6
- anthropic 0.76.0
- openai 2.15.0
- ffmpeg-python 0.2.0
- jinja2 3.1.6
- Deno 2.6.5 (system)
- FFmpeg 6.0 (system)
