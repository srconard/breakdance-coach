# Breakdance Tutorial GIF Generator

## Overview
A Python tool that takes a YouTube breakdancing tutorial video and uses AI to:
1. Analyze the video visually using Gemini to identify tutorial steps and timestamps
2. Generate polished descriptions using a configurable LLM (OpenAI, Anthropic, or Google)
3. Convert each step into a GIF with the description overlaid

## Architecture

```
YouTube URL
    ↓
[yt-dlp] Download video
    ↓
[ffmpeg] Apply cost reduction settings (optional):
         - Downscale to 480p
         - Reduce frame rate to 15fps
         - Trim intro/outro
    ↓
[Gemini API] Upload & analyze video → identify steps with timestamps
    ↓
[Configurable LLM] Generate polished descriptions for each step
(OpenAI / Anthropic / Google)
    ↓
[ffmpeg] Extract video segments → convert to GIFs
    ↓
[Output] Obsidian-compatible Markdown file with embedded GIFs + descriptions
```

## Tech Stack
- **Python 3.10+** - Main language
- **yt-dlp** - YouTube video downloading
- **google-generativeai** - Gemini API for video analysis
- **anthropic** - Claude API for descriptions (optional)
- **openai** - OpenAI API for descriptions (optional)
- **ffmpeg-python** - Video processing and GIF creation
- **Jinja2** - Markdown template generation

## Project Structure
```
Breakdance Coach/
├── src/
│   ├── __init__.py
│   ├── main.py           # CLI entry point
│   ├── downloader.py     # YouTube download logic
│   ├── video_prep.py     # Video preprocessing (resize, trim, fps)
│   ├── video_analyzer.py # Gemini video analysis
│   ├── description.py    # Multi-provider LLM descriptions
│   ├── gif_creator.py    # ffmpeg GIF generation
│   └── output.py         # Markdown generation
├── templates/
│   └── tutorial.md       # Jinja2 template for Obsidian markdown output
├── output/               # Generated tutorials go here
├── config.py             # Settings and API key management
├── requirements.txt
└── README.md
```

## Implementation Steps

### Step 1: Project Setup
- Create directory structure
- Set up requirements.txt with dependencies
- Create config.py for settings and API key management

### Step 2: Video Downloader (`downloader.py`)
- Use yt-dlp to download video from YouTube URL
- Download video WITH audio (Gemini analyzes both visual + verbal cues)
- Download best quality MP4 format with audio track
- Return path to downloaded video file

### Step 3: Video Preprocessor (`video_prep.py`)
Cost reduction settings (all toggleable):
- **Downscale resolution**: Reduce to 480p (default: ON)
- **Reduce frame rate**: Convert to 15fps (default: ON)
- **Trim intro**: Remove N seconds from start (default: 0)
- **Trim outro**: Remove N seconds from end (default: 0)
- **Preserves audio track** (Gemini uses audio for context)
- Uses ffmpeg for all transformations
- Returns path to processed video

### Step 4: Video Analyzer (`video_analyzer.py`)
- Upload processed video to Gemini API
- Wait for processing completion
- Prompt Gemini to identify:
  - Distinct tutorial steps/moves
  - Start/end timestamps for each (MM:SS format)
  - Brief label for each step
- Parse JSON response
- Return structured list of steps with timestamps

### Step 5: Description Generator (`description.py`)
Multi-provider LLM support for writing polished descriptions:
- **Provider interface**: Common base class for all providers
- **Supported providers**:
  - `google` - Gemini Pro / Flash
  - `anthropic` - Claude Sonnet / Haiku
  - `openai` - GPT-4o / GPT-4o-mini
- Takes step labels from video analysis
- Generates engaging, instructional descriptions
- Configurable via CLI flag: `--description-model`

### Step 6: GIF Creator (`gif_creator.py`)
- Use ffmpeg to extract video segments at identified timestamps
- Convert each segment to optimized GIF
- Settings: frame rate, size, color palette optimization
- Return paths to generated GIFs

### Step 7: Output Generator (`output.py`)
- Load Jinja2 Markdown template
- Populate with GIFs, descriptions, and timestamps
- Generate Obsidian-compatible Markdown file
- Use relative paths for GIF embedding: `![[step_1.gif]]` or `![](./gifs/step_1.gif)`
- Copy GIFs to output folder (organized for Obsidian vault)

### Step 8: CLI Interface (`main.py`)
```
python -m src.main <youtube_url> [options]

Options:
  --output, -o          Output folder name (default: video title)
  --description-model   LLM for descriptions: google/anthropic/openai (default: google)
  --no-downscale        Keep original resolution
  --no-fps-reduce       Keep original frame rate
  --trim-intro N        Trim N seconds from start
  --trim-outro N        Trim N seconds from end
  --gif-fps N           GIF frame rate (default: 10)
  --gif-width N         GIF width in pixels (default: 480)
```

## Configuration

### Environment Variables
```
GOOGLE_API_KEY=xxx      # Required - Gemini for video analysis
ANTHROPIC_API_KEY=xxx   # Optional - Claude for descriptions
OPENAI_API_KEY=xxx      # Optional - OpenAI for descriptions
```

### config.py Settings
```python
DEFAULT_SETTINGS = {
    "downscale": True,          # Reduce to 480p
    "downscale_height": 480,
    "reduce_fps": True,         # Reduce to 15fps
    "reduced_fps": 15,
    "trim_intro": 0,            # Seconds to trim from start
    "trim_outro": 0,            # Seconds to trim from end
    "gif_fps": 10,              # Output GIF frame rate
    "gif_width": 480,           # Output GIF width
    "description_model": "google"  # google/anthropic/openai
}
```

## Usage Examples

Basic usage:
```bash
python -m src.main "https://youtube.com/watch?v=..."
```

With options:
```bash
# Use Claude for descriptions, trim 10s intro
python -m src.main "https://youtube.com/watch?v=..." \
  --description-model anthropic \
  --trim-intro 10

# Full quality, no compression
python -m src.main "https://youtube.com/watch?v=..." \
  --no-downscale --no-fps-reduce

# Custom output location
python -m src.main "https://youtube.com/watch?v=..." \
  --output "6-step-toprock"
```

## Output Example
Obsidian Markdown file:
```markdown
# 6-Step Toprock Tutorial

Source: [YouTube Link](https://youtube.com/watch?v=...)

---

## Step 1: Starting Position
**0:00 - 0:15**

![[step_1_starting_position.gif]]

Start in a standing position with feet shoulder-width apart. Keep your weight centered and arms relaxed at your sides.

---

## Step 2: Drop to Floor
**0:15 - 0:32**

![[step_2_drop_to_floor.gif]]

Lower yourself to the floor with your hands first. Plant your palms firmly before transferring weight.

---
```

## Verification
1. Run tool with a sample breakdancing tutorial URL
2. Verify video downloads and preprocesses correctly
3. Verify Gemini identifies reasonable steps/timestamps
4. Test each description provider (google, anthropic, openai)
5. Confirm GIFs are generated at correct timestamps
6. Open Markdown output in Obsidian and verify GIFs render correctly
7. Test cost reduction settings (trim, downscale, fps)

## Dependencies
```
yt-dlp
google-generativeai
anthropic
openai
ffmpeg-python
jinja2
```

## System Requirements
- **ffmpeg** must be installed on the system (not just Python package)
  - Windows: `winget install ffmpeg` or download from ffmpeg.org
  - Mac: `brew install ffmpeg`
  - Linux: `apt install ffmpeg`

## Notes
- Gemini 1.5 Flash is used for video analysis (cheapest multimodal option)
- GIF optimization uses palette generation for smaller file sizes
- Video preprocessing significantly reduces Gemini API costs
- Output uses Obsidian wiki-link syntax `![[file.gif]]` for embedded GIFs
- GIFs stored in subfolder for clean vault organization
- Consider adding manual timestamp override feature later
- Consider adding YouTube transcript as supplementary context later
