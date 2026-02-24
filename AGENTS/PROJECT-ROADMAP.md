# Breakdance Tutorial GIF Generator - Project Roadmap

**Last Updated:** January 24, 2026

## Current Status: 🟡 Working - Refinements Needed

### Completion Summary
| Component | Status | Notes |
|-----------|--------|-------|
| Project Structure | ✅ Complete | All files created |
| Video Downloader | ⚠️ Blocked | YouTube throttling; use local files |
| **Local File Support** | ✅ Tested | `--local-file` flag works |
| Deno JS Runtime | ✅ Installed | Working |
| Video Preprocessor | ✅ Tested | 166MB → 49MB (71% reduction) |
| Gemini Analyzer | ✅ Tested | Correctly identifies 14-15 steps |
| Description Generator | ✅ Tested | Works with rate limiting |
| GIF Creator | ✅ Tested | Uses original video for quality |
| Output Generator | ✅ Tested | Obsidian-compatible markdown |
| CLI Interface | ✅ Complete | All options working |
| Documentation | ✅ Complete | README, CLAUDE.md, AGENTS/ |

---

## Latest Test Results (Jan 24, 2026)

### Full Pipeline Success!
- **Input:** 166MB flare tutorial video
- **Output:** 14 high-quality GIFs + Obsidian markdown
- **Total GIF size:** ~236 MB (high quality) or ~126 MB (standard)

### Sample Output
```
output/flare_tutorial_hq/
├── Flare_Tutorial_by_Sambo.md
└── gifs/
    ├── step_01_flair_demonstration.gif (5.7 MB)
    ├── step_02_introduction_&_prerequisites.gif (20.9 MB)
    └── ... (14 GIFs total)
```

---

## How to Run

```bash
# Set API key
set GOOGLE_API_KEY=your_google_api_key_here

# Run with local file
python -m src.main --local-file "downloads/BEST FLARE TUTORIAL (2020) - BY SAMBO - HOW TO BREAKDANCE (#5).mp4" --title "Flare Tutorial by Sambo"

# With options
python -m src.main --local-file "video.mp4" --title "Tutorial" --gif-fps 10 --gif-width 480
```

---

## Recent Changes (Jan 24, 2026)

| Change | File | Details |
|--------|------|---------|
| Use original video for GIFs | `src/main.py` | Better quality output |
| Increased default GIF quality | `src/main.py` | 640px width, 12fps (was 480px, 10fps) |
| Added timestamp adjustment | `src/main.py` | Handles --trim-intro correctly |
| Fixed API key config | `config.py` | Uses env var `GOOGLE_API_KEY` |
| Updated Gemini model | `video_analyzer.py`, `description.py` | `gemini-exp-1206` |
| Added rate limiting | `description.py` | 15s delay between API calls |

---

## What's Working

### Full Pipeline
1. ✅ Local video loading
2. ✅ Video preprocessing (for Gemini analysis)
3. ✅ Gemini video analysis (step identification)
4. ✅ Description generation (with rate limiting)
5. ✅ GIF/MP4/WebM creation (from original video)
6. ✅ Markdown output (Obsidian-compatible)

### Output Formats
| Format | Size (14 steps) | Playback Controls |
|--------|-----------------|-------------------|
| GIF | ~236 MB | None (auto-loop) |
| MP4 | ~29 MB | Full (scrub, pause) |
| WebM | Smallest | Full |

### Dependencies
```
yt-dlp ✅
google-generativeai ✅ (deprecated but functional)
anthropic ✅
openai ✅
ffmpeg-python ✅
jinja2 ✅
Deno 2.6.5 ✅
FFmpeg 6.0 ✅
```

---

## Open Tasks

### 1. 🔴 Refine Gemini Prompt for Better Step Detection
**Priority:** High

**Current Issue:** The prompt identifies all sections of the video, including parts where the instructor is just talking without visual demonstration.

**Goal:** Only extract steps that have meaningful visual components - actual move demonstrations, not talking-head segments.

**Changes Needed:**
- Update prompt in `src/video_analyzer.py` (lines 82-102)
- Instruct Gemini to skip sections with no visual movement/demonstration
- Have Gemini output fuller, more detailed tutorial descriptions
- Focus on action-oriented segments only

**Example of what to SKIP:**
- Instructor talking to camera (intro, tips sections)
- Static shots with voiceover

**Example of what to KEEP:**
- Actual move demonstrations
- Step-by-step physical instructions
- Slow-motion breakdowns

### 2. 🔴 Fix YouTube Download Feature
**Priority:** High

**Current Issue:** YouTube is actively throttling/blocking yt-dlp requests. Downloads fail after ~290KB with "0 bytes read" errors.

**Possible Solutions:**
1. Use browser cookies (`--cookies-from-browser chrome/edge`)
2. Try OAuth authentication
3. Use alternative download methods (pytube, youtube-dl)
4. Implement PO token generation
5. Wait for yt-dlp updates that bypass new restrictions

**File:** `src/downloader.py`

---

## What's Blocked

### YouTube Downloads
**Status:** Blocked by YouTube throttling (not a code issue)
**Workaround:** Use `--local-file` with manually downloaded videos

### Gemini API Quota (Free Tier)
**Status:** 20 requests/day limit per model
**Workaround:** Wait for daily reset or enable billing

---

## Future Enhancements

### Phase 2: Improved Step Detection
- [ ] Use YouTube transcript as supplementary context
- [ ] Add manual timestamp override option
- [ ] Allow users to review/edit steps before GIF generation

### Phase 3: Better Output Options
- [ ] HTML output option
- [ ] Video output (MP4 with text overlays)
- [ ] Configurable GIF text overlay

### Phase 4: Frame Interpolation (Slow-Mo Scrubbing)
**Priority:** Low

- [ ] Add `--smooth` or `--interpolate` flag to generate high-framerate videos
- [ ] Use FFmpeg `minterpolate` filter (optical flow) or AI-based tools (RIFE)
- [ ] Allows super slow-mo scrubbing to study fast moves frame-by-frame
- [ ] Consider extracting as standalone utility script for reuse

**Implementation Options:**
1. FFmpeg minterpolate (built-in, no dependencies): `minterpolate=fps=60:mi_mode=mci`
2. RIFE (AI-based, higher quality): Requires separate install

### Phase 5: User Experience
- [ ] Web interface (Streamlit/Gradio)
- [ ] Progress bar with ETA
- [ ] Preview mode
- [ ] Batch processing

---

## Technical Debt

1. Migrate from `google.generativeai` to `google.genai`
2. Add proper retry logic with exponential backoff
3. Add unit tests
4. Remove hardcoded Deno path (use PATH discovery)
5. Add input validation for URLs

---

## Long-Term Vision: Utilities Library

Eventually, reusable components from this project should be extracted into a central utilities folder that can be used across projects:

**Proposed Structure:**
```
Workspace/
├── SecondBrain/          # Obsidian vault (notes)
├── Projects/             # Full projects
│   └── BreakdanceCoach/  # This project
└── Utilities/            # Standalone reusable scripts
    ├── video-smoother/   # Frame interpolation
    ├── gif-creator/      # GIF extraction from video
    ├── video-analyzer/   # AI-based video analysis
    └── ...
```

**Candidates for extraction:**
- `gif_creator.py` → Standalone GIF/video segment extractor
- Frame interpolation (when built) → Video smoother utility
- `video_prep.py` → Video preprocessing utility

---

## Key File Locations

| What | Where |
|------|-------|
| Main CLI | `src/main.py` |
| Deno path config | `src/downloader.py` lines 12-14 |
| API key env var | `GOOGLE_API_KEY` |
| Output template | `templates/tutorial.md` |
| Generated tutorials | `output/` folder |
| Session logs | `AGENTS/session-log-*.md` |
