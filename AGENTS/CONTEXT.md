# AI Agent Context - Breakdance Coach Project

**Last Updated:** February 23, 2026

## Quick Status

🟢 **Full pipeline working.** YouTube downloads intermittent. Frame interpolation tool available.

```bash
# Generate tutorial
python -m src.main --local-file "video.mp4" --title "Tutorial" --format mp4

# Slow-mo a clip
python -m src.interpolate "clip.mp4" --slowdown 3
```

## Project Purpose

A tool to help learn breakdancing by converting YouTube tutorial videos into digestible, step-by-step GIF/video tutorials with descriptions. Output is designed for Obsidian note-taking.

## Architecture

```
Video Input (local file or YouTube URL)
    ↓
[yt-dlp] Download (android_vr/tv client workaround)
    ↓
[ffmpeg] Preprocess (480p, 15fps, trim)
    ↓
[Gemini 2.5 Flash] Analyze video → steps + timestamps
    ↓
[LLM] Generate descriptions (Google/Anthropic/OpenAI)
    ↓
[ffmpeg] Create GIFs/MP4/WebM from ORIGINAL video
    ↓
[Jinja2] Obsidian Markdown output
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Gemini 2.5 Flash for video analysis | Native video upload, cost effective |
| Multi-provider descriptions | User wants to compare LLMs |
| Obsidian Markdown output | User's note-taking system |
| MP4 recommended over GIF | 8x smaller, playback controls |
| Cost reduction defaults | 480p + 15fps reduces API costs |
| Frame interpolation as standalone tool | Used on-demand for specific clips |

## Critical File Locations

| What | Where | Notes |
|------|-------|-------|
| CLI entry | `src/main.py` | Orchestrates pipeline |
| Gemini analysis | `src/video_analyzer.py` | `gemini-2.5-flash` |
| Description gen | `src/description.py` | `gemini-2.5-flash` |
| Frame interpolation | `src/interpolate.py` | Standalone slow-mo tool |
| Core dataclass | `src/video_analyzer.py` | `TutorialStep` |
| Output template | `templates/tutorial.md` | Jinja2 |
| API key | `GOOGLE_API_KEY` env var | Never hardcoded |

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

### Frame Interpolation Pattern
```python
# Uses FFmpeg minterpolate filter (optical flow)
# slowdown=3 at fps=60 → generates 180fps interpolated then outputs at 60fps
python -m src.interpolate "clip.mp4" --slowdown 3 --fps 60
```

## What's Been Tested

| Component | Tested | Notes |
|-----------|--------|-------|
| Dependencies install | ✅ | All packages working |
| YouTube download | ✅ | Works with android_vr/tv clients |
| Full pipeline (local) | ✅ | 14-step and 18-step tutorials generated |
| Full pipeline (YouTube) | ✅ | FLARE Workout video downloaded and processed |
| Frame interpolation | ✅ | FFmpeg minterpolate working |

## Dependencies

All installed and working:
- yt-dlp (latest)
- google-generativeai (deprecated but functional)
- anthropic
- openai
- ffmpeg-python
- jinja2
- FFmpeg 6.0 (system)
