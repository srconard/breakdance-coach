# Breakdance Tutorial GIF Generator

Convert YouTube breakdancing tutorials into step-by-step GIFs with AI-generated descriptions. Outputs Obsidian-compatible Markdown files.

## Features

- 🎥 **Video Analysis** - Gemini AI watches the video and identifies distinct tutorial steps
- 📝 **Smart Descriptions** - Choose between Google, Anthropic, or OpenAI for generating step descriptions
- 🎞️ **Multi-Format Output** - GIF, MP4, or WebM clips from the original high-quality video
- 📓 **Obsidian Ready** - Output as Markdown with embedded media and YouTube source link
- 💰 **Cost Efficient** - Video preprocessing reduces API costs (Gemini sees 480p/15fps, clips use original)
- 📁 **Local File Support** - Use pre-downloaded videos with `--local-file`
- 🔄 **Re-clip Tool** - Re-extract clips at higher quality without re-running AI analysis
- 🎬 **RIFE Slow-Motion** - Neural frame interpolation on cloud GPU (Modal.com) for smooth slow-mo
- 📊 **Metadata Export** - JSON metadata saved alongside markdown for later re-processing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
set GOOGLE_API_KEY=your_google_api_key_here   # Windows
export GOOGLE_API_KEY=your_google_api_key_here  # Mac/Linux

# Generate a tutorial from a local video
python -m src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# Generate from YouTube
python -m src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4
```

## Requirements

### System Requirements
- Python 3.10+
- FFmpeg (must be installed on system)
- Deno (recommended for YouTube downloads)

### Install FFmpeg
```bash
# Windows
winget install ffmpeg

# Mac
brew install ffmpeg

# Linux
apt install ffmpeg
```

### Install Deno (for YouTube downloads)
```bash
# Windows
winget install DenoLand.Deno

# Mac/Linux
curl -fsSL https://deno.land/install.sh | sh
```

## Installation

1. Clone or download this repository

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:

```bash
# Required: Google Gemini (video analysis + descriptions)
set GOOGLE_API_KEY=your_google_api_key  # Windows
export GOOGLE_API_KEY=your_google_api_key  # Mac/Linux

# Optional: For RIFE frame interpolation on cloud GPU
python -m modal setup  # One-time browser auth
```

## Usage

### Generate a Tutorial

```bash
# From a local video (recommended)
python -m src.main --local-file "video.mp4" --title "Tutorial Name" --format mp4

# From YouTube
python -m src.main "https://youtube.com/watch?v=VIDEO_ID" --format mp4

# With options
python -m src.main --local-file "video.mp4" --title "Flare Tutorial" \
  --trim-intro 10 --format mp4 --description-model anthropic

# Store YouTube URL when using local file
python -m src.main --local-file "video.mp4" --title "Tutorial" \
  --source-url "https://youtube.com/watch?v=VIDEO_ID"
```

### Re-clip at Higher Quality

After generating a tutorial, you can re-extract clips at higher quality using the saved metadata:

```bash
# Re-clip a specific step at native fps & resolution
python -m src.reclip "output/My_Tutorial" --step 11

# Re-clip all steps
python -m src.reclip "output/My_Tutorial" --all

# Download 1080p from YouTube and re-clip
python -m src.reclip "output/My_Tutorial" --step 11 --download-hq 1080p

# Download best available quality
python -m src.reclip "output/My_Tutorial" --all --download-hq best

# Use a different local video file
python -m src.reclip "output/My_Tutorial" --step 11 --video "hd_video.mp4"
```

### Slow-Motion with Frame Interpolation

```bash
# RIFE (high quality, requires Modal.com account)
python -m src.interpolate "clip.mp4" --slowdown 3 --backend rife

# FFmpeg (local, no GPU needed, lower quality)
python -m src.interpolate "clip.mp4" --slowdown 3 --backend ffmpeg
```

### All Tutorial Generator Options
```
python -m src.main [url] [options]

Input (one required):
  url                     YouTube video URL
  --local-file PATH       Use a local video file instead
  --title NAME            Video title (required with --local-file)
  --source-url URL        YouTube URL to store in metadata (with --local-file)

Output:
  -o, --output NAME       Output folder name (default: video title)
  --format {gif,mp4,webm} Output format (default: gif)
  --fps N                 Output frame rate (default: 12)
  --width N               Output width in pixels (default: 640)

Processing:
  --description-model     LLM for descriptions: google/anthropic/openai (default: google)
  --no-downscale          Keep original video resolution for analysis
  --no-fps-reduce         Keep original frame rate for analysis
  --trim-intro N          Trim N seconds from start
  --trim-outro N          Trim N seconds from end
  --keep-temp             Keep temporary files
```

## Output

The tool generates:
```
output/
└── Tutorial_Name/
    ├── Tutorial_Name.md            # Obsidian markdown file
    ├── tutorial_metadata.json      # Timestamps & settings for re-clipping
    └── gifs/
        ├── step_01_basic_stance.mp4
        ├── step_02_first_move.mp4
        └── ...
```

### Metadata JSON

The `tutorial_metadata.json` file stores all step timestamps, descriptions, source URL, and clip settings. This enables:
- Re-clipping at different quality levels without re-running AI analysis
- Downloading higher quality source video and re-extracting clips
- Programmatic access to tutorial step data

## How It Works

```
Video Input → [Preprocess 480p/15fps] → [Gemini Analysis] → Steps with timestamps
                                                                    ↓
                              Obsidian Markdown ← [Descriptions] ← [Clips from original video]
```

**Key insight:** Gemini sees a low-quality preprocessed video (saves tokens and cost), but clips are extracted from the original full-quality video.

## API Keys Required

| Provider | Required For | Get Key At |
|----------|-------------|------------|
| Google (Gemini) | Video analysis (required) | [Google AI Studio](https://aistudio.google.com/apikey) |
| Anthropic | Descriptions (optional) | [Anthropic Console](https://console.anthropic.com/) |
| OpenAI | Descriptions (optional) | [OpenAI Platform](https://platform.openai.com/api-keys) |
| Modal.com | RIFE interpolation (optional) | [Modal.com](https://modal.com/) |

## Troubleshooting

### YouTube Download Fails with 403 Error
YouTube requires additional authentication. Try:
1. Install Deno: `winget install DenoLand.Deno`
2. Close your browser and retry
3. See [yt-dlp PO Token Guide](https://github.com/yt-dlp/yt-dlp/wiki/PO-Token-Guide)

### "No supported JavaScript runtime" Warning
Install Deno: `winget install DenoLand.Deno`

### GIFs Not Rendering in Obsidian
Ensure GIFs are in the same vault as the markdown file, or use absolute paths.

### RIFE Interpolation Errors
1. Run `python -m modal setup` to authenticate
2. Ensure you have a Modal.com account
3. First run will take a few minutes to build the container image

## Project Structure
```
Breakdance Coach/
├── src/
│   ├── main.py           # Tutorial generator CLI
│   ├── reclip.py          # Re-clip tool (HQ extraction)
│   ├── interpolate.py     # Frame interpolation (RIFE + FFmpeg)
│   ├── rife_modal.py      # RIFE on Modal.com (cloud GPU)
│   ├── downloader.py      # YouTube download (multi-quality)
│   ├── video_prep.py      # Video preprocessing
│   ├── video_analyzer.py  # Gemini analysis
│   ├── description.py     # Multi-LLM descriptions
│   ├── gif_creator.py     # GIF/MP4/WebM creation
│   └── output.py          # Markdown + metadata output
├── templates/
│   └── tutorial.md        # Jinja2 output template
├── config.py              # Settings
├── requirements.txt
└── README.md
```

## License

MIT License - Feel free to use and modify.

## Contributing

This is a personal project for learning breakdancing. Contributions welcome!
