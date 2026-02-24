# Breakdance Tutorial GIF Generator

Convert YouTube breakdancing tutorials into step-by-step GIFs with AI-generated descriptions. Outputs Obsidian-compatible Markdown files.

## Features

- 🎥 **Video Analysis** - Gemini AI watches the video and identifies distinct tutorial steps
- 📝 **Smart Descriptions** - Choose between Google, Anthropic, or OpenAI for generating step descriptions
- 🎞️ **Optimized GIFs** - Two-pass encoding for high quality, small file sizes
- 📓 **Obsidian Ready** - Output as Markdown with embedded GIFs
- 💰 **Cost Efficient** - Video preprocessing reduces API costs
- 📁 **Local File Support** - Use pre-downloaded videos with `--local-file`

## Current Status

⚠️ **YouTube downloads are currently blocked** due to aggressive throttling. Use local video files instead:

```bash
python -m src.main --local-file "your_video.mp4" --title "Tutorial Name"
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

3. Set up API keys (choose one method):

**Option A: Environment Variables (Recommended)**
```bash
# Windows
set GOOGLE_API_KEY=your_google_api_key

# Mac/Linux
export GOOGLE_API_KEY=your_google_api_key
```

**Option B: Edit config.py**
Edit `config.py` and add your API key directly (not recommended for shared environments).

## Usage

### Using Local Video Files (Recommended)

YouTube downloads are currently blocked. Download your video manually and use:

```bash
python -m src.main --local-file "path/to/video.mp4" --title "Tutorial Name"
```

### Using YouTube URLs

```bash
python -m src.main "https://www.youtube.com/watch?v=VIDEO_ID"
```

*Note: YouTube downloads may fail due to throttling. Use local files as a workaround.*

### With Options
```bash
# Local file with options
python -m src.main --local-file "video.mp4" --title "Flare Tutorial" \
  --trim-intro 10 --description-model anthropic

# YouTube URL with options (if working)
python -m src.main "https://youtube.com/watch?v=..." --trim-intro 10

# Full quality (no preprocessing)
python -m src.main --local-file "video.mp4" --title "Tutorial" --no-downscale --no-fps-reduce
```

### All Options
```
python -m src.main [url] [options]

Input (one required):
  url                     YouTube video URL
  --local-file PATH       Use a local video file instead
  --title NAME            Video title (required with --local-file)

Options:
  -o, --output NAME       Output folder name (default: video title)
  --description-model     LLM for descriptions: google/anthropic/openai (default: google)
  --no-downscale          Keep original video resolution
  --no-fps-reduce         Keep original frame rate
  --trim-intro N          Trim N seconds from start
  --trim-outro N          Trim N seconds from end
  --gif-fps N             GIF frame rate (default: 10)
  --gif-width N           GIF width in pixels (default: 480)
  --keep-temp             Keep temporary files
```

## Output

The tool generates:
```
output/
└── Tutorial_Name/
    ├── Tutorial_Name.md     # Obsidian markdown file
    └── gifs/
        ├── step_01_basic_stance.gif
        ├── step_02_first_move.gif
        └── ...
```

### Example Markdown Output
```markdown
# Beginner Toprock Tutorial

Source: [YouTube Video](https://youtube.com/watch?v=...)

---

## Step 1: Basic Stance
**0:15 - 0:45**

![[gifs/step_01_basic_stance.gif]]

Stand with feet shoulder-width apart, knees slightly bent...

---
```

## API Keys Required

| Provider | Required For | Get Key At |
|----------|-------------|------------|
| Google (Gemini) | Video analysis (required) | [Google AI Studio](https://aistudio.google.com/apikey) |
| Anthropic | Descriptions (optional) | [Anthropic Console](https://console.anthropic.com/) |
| OpenAI | Descriptions (optional) | [OpenAI Platform](https://platform.openai.com/api-keys) |

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

## Project Structure
```
Breakdance Coach/
├── src/
│   ├── main.py           # CLI entry point
│   ├── downloader.py     # YouTube download
│   ├── video_prep.py     # Video preprocessing
│   ├── video_analyzer.py # Gemini analysis
│   ├── description.py    # Multi-LLM descriptions
│   ├── gif_creator.py    # GIF generation
│   └── output.py         # Markdown output
├── templates/
│   └── tutorial.md       # Output template
├── config.py             # Settings
├── requirements.txt
└── README.md
```

## License

MIT License - Feel free to use and modify.

## Contributing

This is a personal project for learning breakdancing. Contributions welcome!
