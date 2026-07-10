"""Standalone Instagram movement-library ingest runner for a Synology NAS.

Copy this file to the NAS and run it as root from cron. It downloads queued
Instagram posts, builds contact sheets, and can optionally tag them with the
Gemini CLI.

Example:
    python3 nas_ingest_runner.py --limit 2 --codex   # tag with codex vision ($0)
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


BASE = "/volume1/docker/AI-Agent-Home/ig-ingest"
QUEUE = os.path.join(BASE, "nas_queue.json")
CLIPS_DIR = os.path.join(BASE, "clips")
SHEETS_DIR = os.path.join(BASE, "sheets")
JAR = os.path.join(BASE, ".secrets", "cookies.txt")
DL_LOG = os.path.join(BASE, "download_log.jsonl")
TAGS_NAS = os.path.join(BASE, "tags_nas.json")
RUN_LOG = "/volume1/docker/AI-Agent-Home/logs/ig-ingest.log"
YTDLP = os.path.join(BASE, "bin", "yt-dlp_linux")
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
GEMINI = "gemini"
# Codex vision lane ($0, ChatGPT sub). On the NAS the Gemini CLI free tier is
# UNSUPPORTED_CLIENT, so codex `exec -i <sheet>` with gpt-5.6-sol is the working
# vision-tagging backend. --codex enables it (preferred over --gemini).
CODEX = os.path.join(os.path.expanduser("~"), ".npm-global", "bin", "codex")
CODEX_MODEL = "gpt-5.6-sol"
CODEX_REASONING = "low"

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov")
AUTH_FAILURE_TERMS = (
    "empty media response", "login required", "log in", "rate-limit",
    "rate limit", "please wait a few minutes", "401", "restricted",
    "requested content is not available", "checkpoint", "challenge_required",
    "not available to everyone",
)
PARTNER_STYLE_TERMS = (
    "salsa", "bachata", "zouk", "kizomba", "swing", "lindy", "tango",
    "ballroom", "partner", "acro", "aerial",
)
ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)
TAG_JSON_RE = re.compile(r'\{\s*"clips".*\}', re.DOTALL)
SHEET_NAME_RE = re.compile(r"^(.+)_dur([0-9]+)s_sheet\.jpg$")
DURATION_RE = re.compile(
    r"Duration:\s*([0-9]+):([0-9]+):([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def utc_now_iso():
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Instagram movement-library ingest on a Synology NAS",
    )
    parser.add_argument(
        "--limit", type=nonnegative_int, default=45,
        help="stop after this many successful downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep", type=nonnegative_float, default=8.0,
        help="minimum seconds between download attempts (default: %(default)s)",
    )
    parser.add_argument(
        "--jitter", type=nonnegative_float, default=4.0,
        help="random seconds added to --sleep (default: %(default)s)",
    )
    parser.add_argument(
        "--base", default=BASE,
        help="NAS ingest base directory; re-derives all base-relative paths",
    )
    parser.add_argument(
        "--gemini", action="store_true",
        help="best-effort tag newly generated contact sheets with Gemini",
    )
    parser.add_argument(
        "--codex", action="store_true",
        help="best-effort tag new sheets with codex vision (gpt-5.6-sol, $0); "
             "preferred over --gemini on the NAS",
    )
    parser.add_argument("--codex-bin", default=None, help="override codex binary path")
    parser.add_argument("--codex-model", default=None, help="override codex vision model")
    parser.add_argument("--queue", default=None, help="override nas_queue.json")
    parser.add_argument("--clips-dir", default=None, help="override clips directory")
    parser.add_argument("--sheets-dir", default=None, help="override sheets directory")
    parser.add_argument("--jar", default=None, help="override cookies.txt")
    parser.add_argument("--ytdlp", default=None, help="override yt-dlp binary")
    parser.add_argument("--ffmpeg", default=None, help="override ffmpeg binary")
    return parser.parse_args(argv)


def derived_paths(args):
    """Apply --base first, followed by individual path overrides."""
    base = os.path.abspath(os.path.expanduser(args.base))
    paths = {
        "base": base,
        "queue": os.path.join(base, "nas_queue.json"),
        "clips_dir": os.path.join(base, "clips"),
        "sheets_dir": os.path.join(base, "sheets"),
        "jar": os.path.join(base, ".secrets", "cookies.txt"),
        "dl_log": os.path.join(base, "download_log.jsonl"),
        "tags_nas": os.path.join(base, "tags_nas.json"),
        "run_log": RUN_LOG,
        "ytdlp": os.path.join(base, "bin", "yt-dlp_linux"),
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
        "gemini": GEMINI,
        "codex": CODEX,
        "codex_model": CODEX_MODEL,
        # NAS /tmp is mounted noexec, which breaks the PyInstaller yt-dlp_linux
        # binary ("failed to map segment from shared object"). Point TMPDIR at an
        # exec-allowed volume1 path for the yt-dlp subprocess.
        "tmp_dir": os.path.join(base, "tmp"),
    }
    for argument, key in (
        (args.queue, "queue"),
        (args.clips_dir, "clips_dir"),
        (args.sheets_dir, "sheets_dir"),
        (args.jar, "jar"),
        (args.ytdlp, "ytdlp"),
        (args.ffmpeg, "ffmpeg"),
        (args.codex_bin, "codex"),
    ):
        if argument is not None:
            paths[key] = os.path.abspath(os.path.expanduser(argument))
    if args.codex_model is not None:
        paths["codex_model"] = args.codex_model
    return paths


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_text(path, text):
    ensure_parent(path)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def log_event(run_log, message):
    """Print an event and make a best effort to append it to the cron log."""
    print(message)
    try:
        append_text(run_log, "[%s] %s\n" % (utc_now_iso(), message))
    except OSError as exc:
        print("could not append run log: %s" % exc, file=sys.stderr)


def append_summary(run_log, timestamp, stats, gemini_enabled):
    lines = [
        "=== %s nas ig-ingest run ===" % timestamp,
        "attempted: %s" % stats.get("attempted", 0),
        "ok: %s" % stats.get("ok", 0),
        "failed: %s" % stats.get("failed", 0),
        "auth_aborted: %s" % str(bool(stats.get("auth_aborted"))).lower(),
        "sheets_made: %s" % stats.get("sheets_made", 0),
    ]
    if gemini_enabled:
        lines.append("tagged: %s" % stats.get("tagged", 0))
    lines.extend([
        "cumulative clips-on-disk: %s" % stats.get("clips_on_disk", 0),
        "remaining queue: %s" % stats.get("remaining_queue", 0),
        "",
    ])
    block = "\n".join(lines) + "\n"
    print(block.rstrip())
    try:
        append_text(run_log, block)
    except OSError as exc:
        print("could not append run summary: %s" % exc, file=sys.stderr)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    """Replace a JSON document after fully writing its temporary file."""
    ensure_parent(path)
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def load_queue(path):
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("posts"), list):
        raise ValueError("queue must be a JSON object containing a posts list")
    return [post for post in value["posts"] if isinstance(post, dict)]


def load_downloaded_shortcodes(path):
    """Read successful saved shortcodes from the append-only JSONL log."""
    shortcodes = set()
    if not os.path.isfile(path):
        return shortcodes
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            if (
                isinstance(entry, dict)
                and entry.get("ok") is True
                and entry.get("shortcode")
            ):
                shortcodes.add(str(entry["shortcode"]))
    return shortcodes


def completed_files(clips_dir):
    """Snapshot completed files directly inside the clip directory."""
    files = set()
    if not os.path.isdir(clips_dir):
        return files
    for name in os.listdir(clips_dir):
        path = os.path.abspath(os.path.join(clips_dir, name))
        if (
            os.path.isfile(path)
            and not name.endswith(".part")
            and not name.endswith(".ytdl")
        ):
            files.add(path)
    return files


def file_stems(clips_dir):
    """Return stems for every file, as the queue's secondary dedup check."""
    stems = set()
    if not os.path.isdir(clips_dir):
        return stems
    for name in os.listdir(clips_dir):
        path = os.path.join(clips_dir, name)
        if os.path.isfile(path):
            stems.add(os.path.splitext(name)[0])
    return stems


def dedup_shortcodes(paths):
    """The successful JSONL entries are authoritative; clip stems supplement them."""
    done = load_downloaded_shortcodes(paths["dl_log"])
    done.update(file_stems(paths["clips_dir"]))
    return done


def append_download_attempt(path, entry):
    ensure_parent(path)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def stderr_sample(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    lines = [line.strip() for line in (value or "").splitlines() if line.strip()]
    return "\n".join(lines[-2:])


def is_auth_failure(error_text):
    lowered = (error_text or "").lower()
    return any(term in lowered for term in AUTH_FAILURE_TERMS)


def executable_path(value):
    """Resolve an absolute/relative executable path or a command from PATH."""
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded) or os.path.dirname(expanded):
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return os.path.abspath(expanded)
        return None
    return shutil.which(expanded)


def download_posts(posts, paths, limit, sleep_seconds, jitter):
    done = dedup_shortcodes(paths)
    seen = set()
    work = []
    for post in posts:
        shortcode = str(post.get("shortcode", "") or "")
        if shortcode in done or shortcode in seen:
            continue
        seen.add(shortcode)
        work.append(post)

    successes = 0
    attempts = 0
    failures = 0
    consecutive_auth_failures = 0
    auth_aborted = False
    max_attempts = limit * 3
    total_possible = min(len(work), max_attempts)
    print(
        "download queue: %d pending, target %d successes, maximum %d attempts"
        % (len(work), limit, max_attempts)
    )

    for index, post in enumerate(work):
        if successes >= limit or attempts >= max_attempts:
            break
        attempts += 1
        shortcode = str(post.get("shortcode", "") or "")
        url = str(post.get("url", "") or "")
        before = completed_files(paths["clips_dir"])
        output_template = os.path.join(paths["clips_dir"], "%(id)s.%(ext)s")
        command = [
            paths["ytdlp"],
            "-S", "res:720",
            "--no-playlist",
            "--restrict-filenames",
            "--cookies", paths["jar"],
            "-o", output_template,
            url,
        ]

        returncode = None
        stderr = ""
        diagnostic_text = ""
        ytdlp_env = dict(os.environ)
        tmp_dir = paths.get("tmp_dir")
        if tmp_dir:
            try:
                os.makedirs(tmp_dir, exist_ok=True)
            except OSError:
                pass
            ytdlp_env["TMPDIR"] = tmp_dir
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                env=ytdlp_env,
            )
            returncode = result.returncode
            stderr = result.stderr or ""
            diagnostic_text = "\n".join((result.stdout or "", stderr))
        except subprocess.TimeoutExpired as exc:
            stderr = stderr_sample(exc.stderr) or "yt-dlp timed out after 300 seconds"
            values = []
            for value in (exc.stdout, exc.stderr):
                if isinstance(value, bytes):
                    values.append(value.decode("utf-8", errors="replace"))
                else:
                    values.append(value or "")
            diagnostic_text = "\n".join(values)
        except OSError as exc:
            stderr = str(exc)
            diagnostic_text = stderr

        after = completed_files(paths["clips_dir"])
        new_files = sorted(after - before, key=lambda item: item.lower())
        same_stem = [
            item for item in new_files
            if os.path.splitext(os.path.basename(item))[0] == shortcode
        ]
        downloaded = same_stem[0] if same_stem else (new_files[0] if new_files else None)
        ok = returncode == 0 and downloaded is not None
        error = None
        if not ok:
            error = stderr_sample(stderr)
            if not error:
                if returncode == 0:
                    error = "yt-dlp returned 0 but created no new file"
                elif returncode is not None:
                    error = "yt-dlp exited with status %s without stderr" % returncode
                else:
                    error = "yt-dlp did not complete"
        auth_failure = (not ok) and is_auth_failure(diagnostic_text or error)
        relative_file = (
            os.path.relpath(downloaded, paths["base"])
            if downloaded is not None else None
        )
        entry = {
            "ts": utc_now_iso(),
            "shortcode": shortcode,
            "url": url,
            "ok": bool(ok),
            "file": relative_file,
            "error": error,
            "auth_failure": bool(auth_failure),
        }
        append_download_attempt(paths["dl_log"], entry)

        if ok:
            successes += 1
            consecutive_auth_failures = 0
            status = "ok"
        else:
            failures += 1
            status = "failed"
            if auth_failure:
                consecutive_auth_failures += 1
            else:
                consecutive_auth_failures = 0
        print(
            "[%d/%d] %s: %s (%d/%d successes)"
            % (attempts, total_possible, shortcode, status, successes, limit)
        )
        if error:
            print("error sample: %s" % error)

        if consecutive_auth_failures >= 3:
            auth_aborted = True
            log_event(
                paths["run_log"],
                "aborting: 3 consecutive auth failures (cookies likely expired)",
            )
            break

        more_work = (
            index + 1 < len(work)
            and successes < limit
            and attempts < max_attempts
        )
        if more_work:
            delay = random.uniform(sleep_seconds, sleep_seconds + jitter)
            if delay > 0:
                print("sleeping %.1fs before the next attempt" % delay)
                time.sleep(delay)

    return {
        "attempted": attempts,
        "ok": successes,
        "failed": failures,
        "auth_aborted": auth_aborted,
    }


def valid_duration(value):
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration <= 0:
        return None
    return duration


def probe_duration_ffprobe(clip, ffprobe):
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                clip,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        value = json.loads(result.stdout)
        return valid_duration(value["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return None


def probe_duration_ffmpeg(clip, ffmpeg):
    try:
        result = subprocess.run(
            [ffmpeg, "-i", clip],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = "\n".join((result.stdout or "", result.stderr or ""))
    found = DURATION_RE.search(text)
    if found is None:
        return None
    hours = float(found.group(1))
    minutes = float(found.group(2))
    seconds = float(found.group(3))
    return valid_duration(hours * 3600 + minutes * 60 + seconds)


def probe_duration(clip, paths):
    ffprobe = executable_path(paths["ffprobe"])
    duration = probe_duration_ffprobe(clip, ffprobe) if ffprobe else None
    if duration is not None:
        return duration
    return probe_duration_ffmpeg(clip, paths["ffmpeg"])


def run_ffmpeg(command, timeout=120):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        error = stderr_sample(exc.stderr)
        raise RuntimeError(error or "ffmpeg timed out after %s seconds" % timeout)
    except OSError as exc:
        raise RuntimeError(str(exc))
    if result.returncode != 0:
        error = stderr_sample(result.stderr)
        raise RuntimeError(error or "ffmpeg exited with status %s" % result.returncode)


def existing_sheet(sheets_dir, stem):
    if not os.path.isdir(sheets_dir):
        return None
    prefix = stem + "_dur"
    suffix = "s_sheet.jpg"
    for name in sorted(os.listdir(sheets_dir)):
        path = os.path.join(sheets_dir, name)
        if os.path.isfile(path) and name.startswith(prefix) and name.endswith(suffix):
            return path
    return None


def video_files(clips_dir):
    clips = []
    if not os.path.isdir(clips_dir):
        return clips
    for name in os.listdir(clips_dir):
        path = os.path.abspath(os.path.join(clips_dir, name))
        extension = os.path.splitext(name)[1].lower()
        if os.path.isfile(path) and extension in VIDEO_EXTENSIONS:
            clips.append(path)
    return sorted(clips, key=lambda item: os.path.basename(item).lower())


def make_contact_sheet(clip, paths):
    duration = probe_duration(clip, paths)
    if duration is None:
        raise ValueError("duration unavailable")
    frame_times = [duration * (index + 0.5) / 9 for index in range(9)]
    stem = os.path.splitext(os.path.basename(clip))[0]
    duration_int = int(round(duration))
    sheet_path = os.path.abspath(os.path.join(
        paths["sheets_dir"],
        "%s_dur%ds_sheet.jpg" % (stem, duration_int),
    ))

    # Keep the frame temp dir on the SAME filesystem as the sheets output.
    # /tmp is a separate tmpfs (noexec) on the NAS, which breaks os.replace with
    # "Invalid cross-device link" and can trip the ffmpeg decoder; use volume1.
    sheet_tmp_parent = paths.get("tmp_dir")
    if sheet_tmp_parent:
        try:
            os.makedirs(sheet_tmp_parent, exist_ok=True)
        except OSError:
            sheet_tmp_parent = None
    with tempfile.TemporaryDirectory(prefix="ig_sheet_", dir=sheet_tmp_parent) as temp_dir:
        for index, frame_time in enumerate(frame_times):
            frame_path = os.path.join(temp_dir, "frame_%d.jpg" % index)
            run_ffmpeg([
                paths["ffmpeg"], "-y",
                "-ss", "%.6f" % frame_time,
                "-i", clip,
                "-frames:v", "1",
                "-q:v", "3",
                "-vf", "scale=480:-2",
                # NAS ffmpeg 4.1.9 has no image2 muxer; force the raw mjpeg muxer
                # (portable: also valid on the PC's ffmpeg 6.0).
                "-f", "mjpeg",
                frame_path,
            ])
        temporary_sheet = os.path.join(temp_dir, "sheet.jpg")
        run_ffmpeg([
            paths["ffmpeg"], "-y",
            "-pattern_type", "sequence",
            "-start_number", "0",
            "-i", os.path.join(temp_dir, "frame_%d.jpg"),
            "-vf", "tile=3x3:padding=4",
            "-frames:v", "1",
            "-f", "mjpeg",
            temporary_sheet,
        ])
        os.replace(temporary_sheet, sheet_path)

    sidecar = {
        "clip": os.path.abspath(clip),
        "duration_s": round(duration, 3),
        "frame_times_s": [round(value, 3) for value in frame_times],
        "sheet": sheet_path,
    }
    sidecar_path = os.path.join(paths["sheets_dir"], stem + "_sheet.json")
    write_json(sidecar_path, sidecar)
    return sheet_path


def build_contact_sheets(paths):
    os.makedirs(paths["sheets_dir"], exist_ok=True)
    made = 0
    clips = video_files(paths["clips_dir"])
    print("building contact sheets for %d video clips" % len(clips))
    for clip in clips:
        stem = os.path.splitext(os.path.basename(clip))[0]
        if existing_sheet(paths["sheets_dir"], stem) is not None:
            continue
        try:
            sheet_path = make_contact_sheet(clip, paths)
            made += 1
            print("sheet made: %s" % os.path.basename(sheet_path))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            if str(exc) == "duration unavailable":
                message = "sheet skipped %s: duration unavailable" % os.path.basename(clip)
            else:
                message = "sheet failed %s: %s" % (os.path.basename(clip), exc)
            log_event(paths["run_log"], message)
    return made


def apply_save_policy(entry):
    entry["save"] = "yes"
    moves = entry.get("moves", [])
    if not isinstance(moves, list):
        moves = []
    moves = [str(move) for move in moves]
    entry["moves"] = moves
    move_priority = any(
        move.startswith("partner:") or move in ("lift", "acro")
        for move in moves
    )
    style = str(entry.get("style", "") or "").lower()
    partner_or_lift = move_priority or any(
        term in style for term in PARTNER_STYLE_TERMS
    )
    entry["partner_or_lift"] = partner_or_lift
    worthiness = entry.get("library_worthiness")
    if partner_or_lift and (
        worthiness is None
        or str(worthiness).strip() == ""
        or str(worthiness).lower() == "no"
    ):
        entry["library_worthiness"] = "maybe"
        prefix = "[partner/lift: auto-bumped to maybe] "
        reason = str(entry.get("worthiness_reason", "") or "")
        if not reason.startswith(prefix):
            entry["worthiness_reason"] = prefix + reason


def tags_document(clips, model_label="vision CLI on Synology NAS"):
    return {
        "_comment": (
            "Movement-library tags generated on the NAS. Every clip is saved; "
            "library_worthiness only gates 3D extraction priority."
        ),
        "model": model_label,
        "cost_usd": 0,
        "clips": clips,
    }


def load_tags(path):
    if not os.path.isfile(path):
        return []
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
        raise ValueError("tags_nas.json must contain a clips list")
    return [dict(entry) for entry in value["clips"] if isinstance(entry, dict)]


def sheet_items(paths):
    items = []
    if not os.path.isdir(paths["sheets_dir"]):
        return items
    for name in sorted(os.listdir(paths["sheets_dir"])):
        found = SHEET_NAME_RE.search(name)
        if found is None:
            continue
        sheet_path = os.path.abspath(os.path.join(paths["sheets_dir"], name))
        if not os.path.isfile(sheet_path):
            continue
        stem = found.group(1)
        duration = valid_duration(found.group(2))
        clip = ""
        sidecar_path = os.path.join(paths["sheets_dir"], stem + "_sheet.json")
        if os.path.isfile(sidecar_path):
            try:
                sidecar = read_json(sidecar_path)
                sidecar_duration = valid_duration(sidecar.get("duration_s"))
                if sidecar_duration is not None:
                    duration = sidecar_duration
                clip = str(sidecar.get("clip", "") or "")
            except (OSError, TypeError, ValueError):
                pass
        items.append({
            "shortcode": stem,
            "sheet": sheet_path,
            "clip": clip,
            "duration_s": duration,
        })
    return items


def build_tag_prompt(batch):
    summaries = [
        {
            "sheet": item["sheet"],
            "shortcode": item["shortcode"],
            "duration_s": item.get("duration_s"),
        }
        for item in batch
    ]
    return """Inspect each listed 3x3 contact sheet and tag its movement.

Breaking vocabulary (use exactly): toprock, uprock, 6-step, footwork-other, cc, swipe, windmill, flare, 1990, 2000, headspin, backspin, halo, baby-freeze, chair-freeze, airchair, elbow-freeze, handstand-freeze, kick-up, transition, power-combo, other

PARTNER DANCING: use partner:<style>, for example partner:salsa, partner:bachata, partner:zouk, partner:kizomba, partner:wcs, partner:lindy, partner:tango, partner:ballroom, or partner:social if unsure.

LIFTS/ACRO: use lift for partner lifts/aerials and acro for acrobatics/acroyoga.

Other movement: use other:<name> (parkour, tricking, tutting, popping, etc.).

For every clip return shortcode (= the supplied stem), moves[], style, description (one line), library_worthiness ("yes", "maybe", or "no"), and worthiness_reason. library_worthiness means fitness as a clean single-move reference for 3D extraction, NOT general quality. Return STRICT JSON only in exactly this shape: {"clips":[...]}. Preserve the input order.

The sheet paths below are absolute. Inspect those image files.

Clips:
%s""" % json.dumps(summaries, ensure_ascii=False, indent=2)


def call_gemini(batch, paths):
    prompt = build_tag_prompt(batch)
    try:
        result = subprocess.run(
            [paths["gemini"], "-p", prompt],
            capture_output=True,
            text=True,
            timeout=1500,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Gemini timed out after 1500 seconds")
    except OSError as exc:
        raise RuntimeError(str(exc))
    if result.returncode != 0:
        error = stderr_sample(result.stderr)
        raise RuntimeError(error or "Gemini exited with status %s" % result.returncode)
    cleaned = ANSI_ESCAPE_RE.sub("", result.stdout or "")
    found = list(TAG_JSON_RE.finditer(cleaned))
    if not found:
        raise ValueError("Gemini output did not contain a clips JSON object")
    response = json.loads(found[-1].group(0))
    clips = response.get("clips")
    if not isinstance(clips, list):
        raise ValueError("Gemini clips response was not a list")
    return [dict(entry) for entry in clips if isinstance(entry, dict)]


def build_codex_tag_prompt(batch):
    """Codex receives sheets as attached images (via -i), in order.

    Unlike the Gemini path (which reads file paths), codex sees the images
    directly, so the prompt maps ATTACHMENT ORDER -> shortcode.
    """
    lines = []
    for index, item in enumerate(batch, start=1):
        duration = item.get("duration_s")
        dur_txt = ("%ss" % int(round(duration))) if duration else "unknown length"
        lines.append("image %d = shortcode %s (%s)" % (index, item["shortcode"], dur_txt))
    mapping = "\n".join(lines)
    return """You are tagging movement clips for a breakdance/movement library. %d 3x3 contact sheets are attached as images, in this exact order (each sheet is 9 time-ordered frames of one clip):
%s

Breaking vocabulary (use exactly): toprock, uprock, 6-step, footwork-other, cc, swipe, windmill, flare, 1990, 2000, headspin, backspin, halo, baby-freeze, chair-freeze, airchair, elbow-freeze, handstand-freeze, kick-up, transition, power-combo, other
PARTNER DANCING: partner:<style> (e.g. partner:salsa, partner:bachata, partner:zouk, partner:kizomba, partner:wcs, partner:lindy, partner:tango, partner:ballroom, or partner:social).
LIFTS/ACRO: lift for partner lifts/aerials, acro for acrobatics/acroyoga.
Other movement: other:<name> (parkour, tricking, tutting, popping, etc.). If a sheet shows NO human movement/dance, use moves ["other:none"] and library_worthiness "no".

For every clip return: shortcode (the supplied stem for that image), moves[], style, description (one line), library_worthiness ("yes"|"maybe"|"no" = fitness as a CLEAN single-move reference for 3D extraction, NOT general quality), worthiness_reason.
Return STRICT JSON ONLY, no prose, exactly: {"clips":[...]} preserving the input order.""" % (
        len(batch), mapping,
    )


def call_codex(batch, paths):
    """Tag a batch by attaching each contact sheet to codex via -i.

    codex reads the prompt from stdin (trailing "-"); the variadic -i must not
    swallow the prompt, so the prompt is piped rather than passed positionally.
    Runs read-only (no workspace writes) at $0 on the ChatGPT subscription.
    """
    prompt = build_codex_tag_prompt(batch)
    command = [
        paths["codex"], "exec",
        "--skip-git-repo-check",
        "-s", "read-only",
        "-m", paths.get("codex_model", CODEX_MODEL),
        "-c", "model_reasoning_effort=%s" % CODEX_REASONING,
    ]
    for item in batch:
        command.extend(["-i", item["sheet"]])
    command.append("-")
    try:
        result = subprocess.run(
            command,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=1500,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("codex timed out after 1500 seconds")
    except OSError as exc:
        raise RuntimeError(str(exc))
    if result.returncode != 0:
        error = stderr_sample(result.stderr) or stderr_sample(result.stdout)
        raise RuntimeError(error or "codex exited with status %s" % result.returncode)
    cleaned = ANSI_ESCAPE_RE.sub("", result.stdout or "")
    found = list(TAG_JSON_RE.finditer(cleaned))
    if not found:
        raise ValueError("codex output did not contain a clips JSON object")
    response = json.loads(found[-1].group(0))
    clips = response.get("clips")
    if not isinstance(clips, list):
        raise ValueError("codex clips response was not a list")
    return [dict(entry) for entry in clips if isinstance(entry, dict)]


def merge_tag_batch(tagged_clips, batch, model_clips, model_label=None):
    positions = {
        str(entry.get("shortcode", "")): index
        for index, entry in enumerate(tagged_clips)
        if entry.get("shortcode")
    }
    model_by_stem = {
        str(entry.get("shortcode", "")): entry
        for entry in model_clips
        if entry.get("shortcode")
    }
    missing = [
        item["shortcode"] for item in batch
        if item["shortcode"] not in model_by_stem
    ]
    if missing:
        raise ValueError("model response omitted %d requested clips" % len(missing))

    for item in batch:
        stem = item["shortcode"]
        current = dict(tagged_clips[positions[stem]]) if stem in positions else {}
        current.update(model_by_stem[stem])
        current["shortcode"] = stem
        current["duration_s"] = item.get("duration_s")
        current["sheet"] = item["sheet"]
        current["clip"] = item.get("clip", "")
        if model_label:
            current["tagged_by"] = model_label
        apply_save_policy(current)
        if stem in positions:
            tagged_clips[positions[stem]] = current
        else:
            positions[stem] = len(tagged_clips)
            tagged_clips.append(current)


TAG_BACKENDS = {
    "codex": (call_codex, "gpt-5.6-sol via codex CLI (NAS, $0)"),
    "gemini": (call_gemini, "Gemini CLI on Synology NAS"),
}


def tag_sheets(paths, backend="codex"):
    call_model, model_label = TAG_BACKENDS[backend]
    try:
        tagged_clips = load_tags(paths["tags_nas"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log_event(paths["run_log"], "%s tagging skipped/failed: %s" % (backend, exc))
        return 0

    for entry in tagged_clips:
        apply_save_policy(entry)
    try:
        write_json(paths["tags_nas"], tags_document(tagged_clips, model_label))
    except OSError as exc:
        log_event(paths["run_log"], "%s tagging skipped/failed: %s" % (backend, exc))
        return 0

    existing = {
        str(entry.get("shortcode", ""))
        for entry in tagged_clips
        if entry.get("shortcode")
    }
    pending = [
        item for item in sheet_items(paths)
        if item["shortcode"] not in existing
    ]
    tagged = 0
    for offset in range(0, len(pending), 12):
        batch = pending[offset:offset + 12]
        try:
            model_clips = call_model(batch, paths)
            merge_tag_batch(tagged_clips, batch, model_clips, model_label)
            write_json(paths["tags_nas"], tags_document(tagged_clips, model_label))
            tagged += len(batch)
            print("%s tagged %d sheets" % (backend, len(batch)))
        except (
            OSError, TypeError, ValueError, KeyError, RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            log_event(paths["run_log"], "%s tagging skipped/failed: %s" % (backend, exc))
    return tagged


def remaining_queue_count(posts, done):
    remaining = 0
    seen = set()
    for post in posts:
        shortcode = str(post.get("shortcode", "") or "")
        if shortcode in seen:
            continue
        seen.add(shortcode)
        if shortcode not in done:
            remaining += 1
    return remaining


def fatal_result(paths, timestamp, stats, gemini_enabled, message):
    log_event(paths["run_log"], "fatal: %s" % message)
    append_summary(paths["run_log"], timestamp, stats, gemini_enabled)
    return 2


def main(argv=None):
    args = parse_args(argv)
    paths = derived_paths(args)
    timestamp = utc_now_iso()
    stats = {
        "attempted": 0,
        "ok": 0,
        "failed": 0,
        "auth_aborted": False,
        "sheets_made": 0,
        "tagged": 0,
        "clips_on_disk": 0,
        "remaining_queue": 0,
    }

    if not os.path.isfile(paths["queue"]):
        return fatal_result(
            paths, timestamp, stats, (args.codex or args.gemini),
            "queue file not found: %s" % paths["queue"],
        )
    try:
        posts = load_queue(paths["queue"])
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return fatal_result(
            paths, timestamp, stats, (args.codex or args.gemini),
            "could not load queue %s: %s" % (paths["queue"], exc),
        )

    resolved_ytdlp = executable_path(paths["ytdlp"])
    if resolved_ytdlp is None:
        stats["remaining_queue"] = len(posts)
        return fatal_result(
            paths, timestamp, stats, (args.codex or args.gemini),
            "yt-dlp binary not found or not executable: %s" % paths["ytdlp"],
        )
    paths["ytdlp"] = resolved_ytdlp

    try:
        os.makedirs(paths["clips_dir"], exist_ok=True)
        os.makedirs(paths["sheets_dir"], exist_ok=True)
        ensure_parent(paths["dl_log"])
    except OSError as exc:
        stats["remaining_queue"] = len(posts)
        return fatal_result(
            paths, timestamp, stats, (args.codex or args.gemini),
            "could not prepare ingest directories: %s" % exc,
        )

    if not os.path.isfile(paths["jar"]):
        log_event(paths["run_log"], "warning: cookies jar not found: %s" % paths["jar"])

    try:
        download_stats = download_posts(
            posts,
            paths,
            args.limit,
            args.sleep,
            args.jitter,
        )
    except OSError as exc:
        stats["remaining_queue"] = len(posts)
        return fatal_result(
            paths, timestamp, stats, (args.codex or args.gemini),
            "download state could not be read or written: %s" % exc,
        )
    stats.update(download_stats)
    try:
        stats["sheets_made"] = build_contact_sheets(paths)
    except OSError as exc:
        log_event(paths["run_log"], "sheet stage skipped/failed: %s" % exc)
    tag_backend = "codex" if args.codex else ("gemini" if args.gemini else None)
    if tag_backend:
        try:
            stats["tagged"] = tag_sheets(paths, tag_backend)
        except OSError as exc:
            log_event(
                paths["run_log"],
                "%s tagging skipped/failed: %s" % (tag_backend, exc),
            )

    done = dedup_shortcodes(paths)
    stats["clips_on_disk"] = len(completed_files(paths["clips_dir"]))
    stats["remaining_queue"] = remaining_queue_count(posts, done)
    append_summary(paths["run_log"], timestamp, stats, bool(tag_backend))
    return 0


if __name__ == "__main__":
    sys.exit(main())
