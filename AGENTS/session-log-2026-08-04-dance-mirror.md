# Session Log — Dance Mirror Slice 1 — August 4, 2026

## Summary

Added Dance Mirror as the monorepo's fourth product surface. It turns a local short-form dance clip, or a URL accepted by yt-dlp, into an editable lesson JSON and one self-contained offline HTML practice page.

## What Was Built

### Package and CLI

- Added `dance_mirror/` with the repository-standard `src/` and `templates/` layout.
- Added `python -m dance_mirror.src.main build` for local-file and URL ingestion.
- Added `python -m dance_mirror.src.main refresh <lesson_dir>` to re-render solely from `dance_lesson.json`; refresh never invokes Gemini.
- Local input is copied into the lesson directory as `source.mp4`. URL downloads use the existing yt-dlp wrapper without calling its YouTube-only ID parser.
- URL failures give a user-facing save-and-use-`--local-file` message and do not attempt to bypass access controls.

### Segmentation and Schema

- Default segmentation uses `gemini-2.5-flash` and follows the existing upload, processing poll, JSON parsing, and remote-file cleanup pattern.
- Gemini is prompted for 3–8 chronological dance phrases with literal movement cues, not invented move names.
- Model ranges are sorted, clamped to the video duration, joined into contiguous boundaries, stripped of zero-length entries, and merged when more than eight are returned.
- `--no-ai`, a missing Google key, or an AI failure produces a deterministic uniform split using `--parts` (default 4).
- Added strict load/save validation for schema version, required fields, 3–8 parts, finite timing/FPS values, and per-part video bounds. Hand-adjusted practice ranges may intentionally overlap adjacent parts.
- Duration comes from the existing video-prep probe; FPS comes from `ffprobe` and supports fractional frame rates.

### Offline Practice Page

- The renderer base64-inlines `source.mp4` and embeds the lesson JSON into one dependency-free HTML file.
- Mobile-first breakdown lists the creator credit, canonical-source link, parts, progress, and cumulative build-the-dance ladder.
- Part practice includes A–B looping, a 3-2-1 count-in, 0.25×/0.5×/0.75×/1× speeds, mirror-on-by-default, ±0.1-second boundary nudges, save-to-part, one-frame stepping, cues, and previous/Got-it progression.
- Build practice advances from Part 1 through cumulative ranges, then full dance at 0.75× and 1×.
- A `requestAnimationFrame` watcher enforces the loop end instead of relying on coarse `timeupdate` events.
- Progress, settings, and in-page lesson edits persist in `localStorage`, keyed by title and source. A fingerprint prevents stale stored lesson data from masking a newly refreshed embedded JSON.
- Export downloads the current in-page state as `dance_lesson.json` for saving beside the source and refreshing.

## Commands Run

```powershell
python -m compileall -q dance_mirror
python -m dance_mirror.src.main --help
python -m dance_mirror.src.main build --help

python -m dance_mirror.src.main build --local-file "downloads\instagram\clips\CyomjspuuEJ.mp4" --title "Tutting geometry" --credit "@mandijuana_" --source-url "https://www.instagram.com/reel/CyomjspuuEJ/" --no-ai --parts 4 -o output/dance_mirror/tutting-demo

python -m dance_mirror.src.main refresh output/dance_mirror/tutting-demo
```

## Demo and Validation

- Generated `output/dance_mirror/tutting-demo/dance_lesson.json`, `source.mp4`, and `lesson.html` from the verified 30-second local clip.
- The initial offline build contained four uniform parts over 30.002 seconds.
- Hand-edited Part 1's end boundary from 7.5 to 7.4 seconds, ran `refresh`, and confirmed 7.4 was embedded in the regenerated HTML.
- Parsed the HTML, confirmed all required control elements, confirmed there are no automatic external URL loads, and decoded the data URI to the exact 4,056,687-byte source video.
- Final self-contained HTML size was 5,436,376 bytes.
- Python compilation, CLI parsing, uniform segmentation, 5,000 randomized repair cases, and over-segmentation merge checks passed.

## Design Decisions

- Kept the JSON authoritative and human-editable instead of adding a server or build-time frontend dependency.
- Used the original clip as the only media asset; all practice modes operate by changing HTML5 video loop bounds.
- Stored a base-lesson fingerprint with local state so a hand edit plus `refresh` reliably wins over stale browser data while normal in-page edits still persist.
- Restricted `--parts` to 3–8, matching the product lesson-size contract.

## Known Gaps

- Gemini segmentation is wired but untested in the offline sandbox.
- URL ingestion is wired but untested in the offline sandbox; platform availability and yt-dlp extractor behavior need a connected smoke test.
- Beat-grid or audio-aware segmentation is deferred; no heavy audio/numeric dependency was added.
- Automated browser navigation to `file://` was blocked by the browser-control security policy, so visual interaction was not automated here. The page received structural, content, data-URI, and external-request sanity checks instead.

## Gemini Client Migration

- Migrated Dance Mirror segmentation from the deprecated `google.generativeai` client to `google.genai`.
- The environment's new-format `AQ.` Gemini key was rejected by the legacy client's discovery call even though a direct REST models request with `x-goog-api-key` returned 200 and listed `gemini-2.5-flash`.
- Dance Mirror now uses the modern client for file upload, processing polling, content generation, and remote-file cleanup. The tutorial generator remains on the legacy client pending its separate migration.
- Intermittent malformed free-text JSON could still send otherwise successful Gemini segmentation through the uniform-split fallback.
- Segmentation now requests a structured JSON response schema and retries content generation once when parsing fails or the response is not an array.

## Live Verification (orchestrator, same evening)

Codex authored all substantive code above (three `codex exec` runs, gpt-5.6-sol). The orchestrating Claude session then verified against the network and a real browser:

- **Offline path:** fresh `build --no-ai` runs clean end-to-end; JSON schema, contiguous bounds, and fps all correct. Hand-edit + `refresh` contract confirmed independently.
- **Gemini path (live):** first runs exposed two real issues — (1) the upload poll exited on "not PROCESSING" and could call generate before the file was ACTIVE (one-line hardening applied directly by the orchestrator: poll until ACTIVE/FAILED); (2) intermittent malformed free-text JSON (`Expecting ',' delimiter`), fixed by Codex with the structured response schema + retry above. After both fixes: **3/3 consecutive live successes** with varied, meaningful breakdowns.
- **Canonical demo regenerated with real AI parts:** `output/dance_mirror/tutting-demo/` — 7 parts ("Setup Pose", "Finger Hinges", "Wrist Hinges", "Quick Wrist Hinges", "Right Angle Shift", "Left Angle Shift", "Concluding Actions") over the 30 s @mandijuana_ tutting clip.
- **Browser QA (Chrome via localhost, 430 px viewport):** breakdown list, credit + canonical-source link, learn view (count-in label, Play loop, Mirror-on default, 4 speed presets, frame step @ 30.00 fps, A–B nudge with save-to-part, Got-it), build-the-dance ladder, and Export button all render and navigate correctly.
- **Environmental finding, NOT a page bug:** video playback in this Chrome instance is currently wedged browser-wide — even Chrome's native player spins forever on a pristine 8 KB ffmpeg test MP4 in a fresh tab (readyState 0 on data:, blob:, and http: sources alike; clip codec is plain H.264 High yuv420p). ffmpeg frame extraction proves the embedded video content is correct. **Restart Chrome, then re-open `lesson.html` to see playback.**
