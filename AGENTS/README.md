# AGENTS Folder

**Last Updated:** January 23, 2026

Documentation and context for AI agents working on this project.

## Current Status: 🟡 Ready for Testing

All code complete. YouTube downloads blocked. **Use local files to test.**

```bash
python -m src.main --local-file "video.mp4" --title "Tutorial Name"
```

## Files

| File | Description |
|------|-------------|
| `CONTEXT.md` | Technical architecture, code patterns, API usage |
| `PROJECT-ROADMAP.md` | Current status, blockers, next steps |
| `session-log-2026-01-22.md` | Detailed build history |
| `initial-breakdance-tutorial-plan-2026-01-22.md` | Original project plan |

## Quick Reference

### What's Working
- ✅ Deno JS runtime installed
- ✅ All Python dependencies
- ✅ Local file support (`--local-file`)
- ✅ FFmpeg 6.0

### What's Blocked
- ❌ YouTube downloads (throttling)
- ⚠️ Full pipeline untested (needs local video)

### Key File Locations
- **Google API key:** `config.py` line 44
- **Deno path:** `src/downloader.py` lines 12-14
- **CLI entry:** `src/main.py`

## For AI Agents

1. Read `CONTEXT.md` for architecture
2. Check `PROJECT-ROADMAP.md` for current blockers
3. YouTube download issues are throttling, not code bugs
4. Test with `--local-file` flag
