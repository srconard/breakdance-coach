# Move Library (web app) and the 3D audit

*2026-11-17.* A phone-first static site that gathers everything this repo has produced, one page per move: his saved Instagram clips (by move, with a "not yet sorted" shelf), his video-to-3D extractions with a clean / usable / broken badge, the motion-capture reference (Mixamo, CMU 85, AIST++, FineDance) beside the moves it shows, tutorial-generator step clips and Dance Mirror lessons.

Served privately by the NAS bridge at `/webapps/move-library/` (tailnet only: it holds other people's videos; never publish it).

## Pipeline

1. **PC** (`pc-run`/`pc-exec` lane, $0):
   - `python -m analyzer_3d.src.library_audit <results> --clips <clips dir> --out <audit dir>`: per extraction, a mannequin GLB, an audit sheet (source frames with the 3D skeleton projected on top + the world-frame 3D with the floor line), and measured numbers (2D fit, floating, sliding, jitter, drift) with a draft grade.
   - `python -m analyzer_3d.src.library_vision_judge <audit dir>`: Codex CLI vision ($0 on the ChatGPT plan) grades each sheet and tags failures.
   - Viewer bundle: `cd viewer && npx esbuild src/app.js --bundle --minify --format=iife --outfile=dist/app.bundle.js`.
2. **Copy to the NAS** (tar over ssh) into `ig-extract/library-assets/` (`audit/`, `viewer/`, mocap GLBs, tutorial clips).
3. **NAS**: `python3 -m analyzer_3d.src.move_library_site --config analyzer_3d/nas_deploy/move_library_site.json`. Pure stdlib (python 3.8). Media are hard-linked from where they already live, so the site adds no disk.
4. **QA**: `node viewer/qa_library.mjs --base http://192.168.0.157:8787/webapps/move-library/ --out qa <pages…>`. It opens each page headless at phone size and reports console errors and failed requests. On viewer pages it also checks that the model loads, the clock runs and the video advances.

## Viewer

`viewer/library.html` + `app.bundle.js` is one shared page: `viewer.html?glb=…&video=…&fps=…&title=…[&t=s][&loop=a,b][&back=page]`.

- `app.js` keeps the self-contained base64 mode for `make_viewer.py`.
- In URL mode it fetches the GLB, and the video as a blob. The bridge's `/webapps/` route has no HTTP Range support, and a streamed `<video>` stalls under the viewer's constant re-seeks.

## Grades

The final badge is the vision verdict, calibrated by the numbers (see `grade_of` in `move_library_site.py`), plus dated per-clip overrides in the config.

The audit write-up and the plan to fix video-to-3D for breaking are in the vault: `AGENTS/dev/breakdance-coach/docs/2026-11-17 move-library-and-3d-audit.md`.
