"""Instagram saved-post ingest pipeline for the movement library.

Parses an Instagram data export, downloads a prioritized sample with yt-dlp,
builds 3x3 contact sheets with ffmpeg, and prepares a VLM tagging queue.
Only the Python standard library is required; yt-dlp, ffmpeg, and ffprobe are
invoked as external executables when their subcommands are used.

Usage:
    python -m analyzer_3d.src.ig_ingest parse
    python -m analyzer_3d.src.ig_ingest download --limit 12
    python -m analyzer_3d.src.ig_ingest sheets
    python -m analyzer_3d.src.ig_ingest queue
    python -m analyzer_3d.src.ig_ingest nas-queue
    python -m analyzer_3d.src.ig_ingest refresh-cookies
    python -m analyzer_3d.src.ig_ingest tag
    python -m analyzer_3d.src.ig_ingest extract-queue
    python -m analyzer_3d.src.ig_ingest nightly

The file also supports direct script execution:
    python analyzer_3d/src/ig_ingest.py parse
"""

import argparse
import base64
import html
import hashlib
import json
import math
import os
import random
import re
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from .vlm_movetag import MOVE_VOCAB
except ImportError:
    # Support ``python analyzer_3d/src/ig_ingest.py ...`` as well as -m.
    _MODULE_DIR = Path(__file__).resolve().parent
    if str(_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(_MODULE_DIR))
    from vlm_movetag import MOVE_VOCAB


DEFAULT_EXPORT_DIR = Path(
    "downloads/instagram/instagram-conardshawn-2026-07-06-TxxS8dfE/"
    "your_instagram_activity/saved"
)
DEFAULT_OUT_DIR = Path("output/ig_ingest")
DEFAULT_CLIPS_DIR = Path("downloads/instagram/clips")
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA")
DEFAULT_COOKIES_JAR = (
    Path(_LOCALAPPDATA) / "breakdance-coach" / "ig_cookies.txt"
    if _LOCALAPPDATA else None
)
DEFAULT_DEBUG_PORT = 9333
BRAVE_EXE_CANDIDATES = [
    Path("C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"),
    Path("C:/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe"),
    Path(os.path.expandvars(
        "%LOCALAPPDATA%/BraveSoftware/Brave-Browser/Application/brave.exe"
    )),
]
REAL_BRAVE_USER_DATA_DIR = (
    Path(_LOCALAPPDATA) / "BraveSoftware" / "Brave-Browser" / "User Data"
    if _LOCALAPPDATA else None
)
MOVEMENT_COLLECTION_KEYWORDS = [
    "parkour", "dance", "bboy", "break", "movement", "tricking",
    "flip", "gym", "acro",
]
MOVEMENT_HINT_TERMS = [
    "dance", "dancer", "dancing", "bboy", "bgirl", "breakin", "breaking",
    "breakdance", "breakdancing", "footwork", "toprock", "powermove",
    "power move", "windmill", "flare", "headspin", "freeze", "cypher",
    "parkour", "freerun", "freerunning", "tricking", "tricks", "tutting",
    "flow", "movement", "calisthenics", "mobility", "handstand", "capoeira",
    "flips", "flip", "backflip", "gymnastics", "acro", "choreography",
    "choreo", "house dance", "popping", "locking", "krump",
    "contemporary dance", "salsa", "bachata", "zouk", "kizomba",
    "west coast swing", "wcs", "lindy", "lindy hop", "swing dance", "tango",
    "ballroom", "partner dance", "partner dancing", "partnering",
    "partner work", "partner", "duo", "contact improv",
    "contact improvisation", "acroyoga", "acro yoga", "acrobatics", "lift",
    "lifts", "aerial", "aerials", "dip", "dips",
]
AUTH_FAILURE_TERMS = [
    "empty media response", "login required", "log in", "rate-limit",
    "rate limit", "please wait a few minutes", "401", "restricted",
    "requested content is not available", "checkpoint", "challenge_required",
    "not available to everyone",
]
PARTNER_STYLE_TERMS = [
    "salsa", "bachata", "zouk", "kizomba", "swing", "lindy", "tango",
    "ballroom", "partner", "acro", "aerial",
]
MONTAGE_MARKERS = [
    "montage", "edit", "edits", "edited", "music video", "compilation",
    "promo", "promotional", "trailer", "recap", "mashup", "showreel",
    "show reel", "highlight reel", "highlights", "many cuts", "fast cuts",
    "fast-cut", "quick cuts", "multiple clips", "several clips",
]
REVIEW_MARKERS = [
    "battle", "battles", "cypher", "cipher", "versus", "vs", "crowd",
    "crew", "jam", "multiple dancers", "two dancers", "several dancers",
    "multiple people",
]
FLOOR_PLANE_ONLY_MARKERS = [
    "parkour", "freerun", "freerunning", "freerunner", "vault", "vaults",
    "tricking", "rooftop", "obstacle", "obstacles", "street workout",
]
TAG_MODEL = "Gemini 3.1 Pro (High)"
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov"}
ANSI_ESCAPE_RE = re.compile(
    r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
)
TAG_JSON_RE = re.compile(r'\{\s*"clips".*\}', re.DOTALL)
MONTAGE_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(marker) for marker in MONTAGE_MARKERS)
    + r")(?!\w)",
    re.IGNORECASE,
)
REVIEW_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(marker) for marker in REVIEW_MARKERS)
    + r")(?!\w)",
    re.IGNORECASE,
)
FLOOR_PLANE_ONLY_RE = re.compile(
    r"(?<!\w)(?:" + "|".join(
        re.escape(marker)
        for marker in FLOOR_PLANE_ONLY_MARKERS
        if marker != "tricking"
    ) + r")(?!\w)",
    re.IGNORECASE,
)
TRICKING_RE = re.compile(r"(?<!\w)tricking(?!\w)", re.IGNORECASE)
BREAKING_RE = re.compile(r"(?<!\w)breaking(?!\w)", re.IGNORECASE)


# Instagram's export is regular enough that narrowly anchored regexes are less
# error-prone than trying to repair its deeply nested tables with HTMLParser.
POST_URL_RE = re.compile(
    r'<a target="_blank" href="'
    r'(https://www\.instagram\.com/(reel|p|tv)/([^/?#"]+)[^"]*)"'
)
SAVED_AT_RE = re.compile(r'<div class="_3-94 _a6-o">(.*?)</div>', re.DOTALL)
CAPTION_RE = re.compile(
    r'<td class="_a6_q">Caption</td>'
    r'<td class="_2piu _a6_r">(.*?)</td>',
    re.DOTALL,
)
HASHTAGS_SECTION_RE = re.compile(
    r'<h2[^>]*>Hashtags</h2>(.*?)(?=<h2[^>]*>Owner</h2>|$)',
    re.DOTALL,
)
HASHTAG_RE = re.compile(r'<div class="_a6-p">([^<]*)</div>')
OWNER_HEADER_RE = re.compile(r'<h2[^>]*>Owner</h2>')
OWNER_ROWS_RE = re.compile(
    r'<tr><td class="_a6_q">Name</td>'
    r'<td class="_2piu _a6_r">([^<]*)</td></tr>'
    r'<tr><td class="_a6_q">Username</td>'
    r'<td class="_2piu _a6_r">([^<]*)</td></tr>'
)
OWNER_USERNAME_RE = re.compile(
    r'<tr><td class="_a6_q">Username</td>'
    r'<td class="_2piu _a6_r">([^<]*)</td></tr>'
)
COLLECTION_HEADER_RE = re.compile(
    r'<td class="_a6_q">Name</td>'
    r'<td class="_2piu _a6_r">([^<]*)</td></tr>'
    r'<tr><td class="_a6_q">Type</td>'
    r'<td class="_2piu _a6_r">([^<]*)</td></tr>'
    r'<tr><td class="_a6_q">Privacy</td>'
)


# ---------------------------------------------------------------- utilities

def utc_now_iso() -> str:
    """Return a timezone-aware ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def extracted_text(value: str) -> str:
    """Unescape and trim one text value extracted from the export HTML."""
    return html.unescape(value).strip()


def write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON using the ingest pipeline's common formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def movement_collection(name: str) -> bool:
    """Return whether a collection name explicitly marks movement content."""
    lowered = name.lower()
    return any(keyword in lowered for keyword in MOVEMENT_COLLECTION_KEYWORDS)


def movement_hints(caption: str, hashtags: list[str]) -> list[str]:
    """Return sorted recall-oriented movement terms found in post text."""
    haystack = (caption + " " + " ".join(hashtags)).lower()
    normalized_hashtags = [
        re.sub(r"[^a-z0-9]", "", hashtag.lower()) for hashtag in hashtags
    ]
    matches = {
        term for term in MOVEMENT_HINT_TERMS
        if term in haystack
        or any(
            re.sub(r"[^a-z0-9]", "", term) in hashtag
            for hashtag in normalized_hashtags
        )
    }
    return sorted(matches)


def parse_saved_at(value: str) -> str | None:
    """Convert an Instagram saved-at stamp to ISO 8601, or None if invalid."""
    try:
        parsed = datetime.strptime(extracted_text(value), "%b %d, %Y %I:%M %p")
    except ValueError:
        return None
    return parsed.isoformat()


def relative_to_cwd(path: Path) -> str:
    """Return a path relative to the current working directory."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return os.path.relpath(resolved, Path.cwd().resolve())


def stderr_sample(stderr: str | bytes | None) -> str:
    """Return the last two non-empty stderr lines from a subprocess."""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    return "\n".join(lines[-2:])


def find_existing_clip(clips_dir: Path, shortcode: str) -> Path | None:
    """Find a completed yt-dlp output whose stem is exactly the shortcode."""
    if not clips_dir.exists():
        return None
    matches = sorted(
        path for path in clips_dir.iterdir()
        if path.is_file()
        and path.stem == shortcode
        and not path.name.endswith((".part", ".ytdl"))
    )
    return matches[0] if matches else None


def completed_files(clips_dir: Path) -> set[Path]:
    """Snapshot completed files immediately inside the clips directory."""
    if not clips_dir.exists():
        return set()
    return {
        path.resolve()
        for path in clips_dir.iterdir()
        if path.is_file() and not path.name.endswith((".part", ".ytdl"))
    }


# ---------------------------------------------------------------- parse

def parse_collections(text: str) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Parse exact collection headers and map media shortcodes to names."""
    headers = list(COLLECTION_HEADER_RE.finditer(text))
    collections = []
    shortcode_collections: dict[str, list[str]] = {}

    for index, match in enumerate(headers):
        block_end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[match.end():block_end]
        name = extracted_text(match.group(1))
        collection_type = extracted_text(match.group(2))
        movement_match = movement_collection(name)

        media_codes = []
        for url_match in POST_URL_RE.finditer(block):
            shortcode = extracted_text(url_match.group(3))
            media_codes.append(shortcode)
            names = shortcode_collections.setdefault(shortcode, [])
            if name not in names:
                names.append(name)

        collections.append({
            "name": name,
            "type": collection_type,
            "media_count": len(media_codes),
            "movement_match": movement_match,
        })

    return collections, shortcode_collections


def parse_post_block(block: str, saved_at_raw: str) -> dict[str, Any] | None:
    """Parse one block terminated by a saved-at div."""
    url_match = POST_URL_RE.search(block)
    if not url_match:
        return None

    caption_match = CAPTION_RE.search(block)
    caption = extracted_text(caption_match.group(1)) if caption_match else ""

    hashtags = []
    hashtags_match = HASHTAGS_SECTION_RE.search(block)
    if hashtags_match:
        hashtags = [
            extracted_text(match.group(1))
            for match in HASHTAG_RE.finditer(hashtags_match.group(1))
            if extracted_text(match.group(1))
        ]

    owner = {"username": "", "name": ""}
    owner_header = OWNER_HEADER_RE.search(block)
    if owner_header:
        owner_match = OWNER_ROWS_RE.search(block, owner_header.end())
        if owner_match:
            owner = {
                "username": extracted_text(owner_match.group(2)),
                "name": extracted_text(owner_match.group(1)),
            }
        else:
            # A few real export rows omit Name entirely but retain Username.
            username_match = OWNER_USERNAME_RE.search(block, owner_header.end())
            if username_match:
                owner["username"] = extracted_text(username_match.group(1))

    return {
        "url": extracted_text(url_match.group(1)),
        "shortcode": extracted_text(url_match.group(3)),
        "kind": url_match.group(2),
        "saved_at": parse_saved_at(saved_at_raw),
        "caption": caption,
        "hashtags": hashtags,
        "owner": owner,
        "collections": [],
        "movement": "unknown",
    }


def parse_posts(text: str) -> tuple[list[dict[str, Any]], int]:
    """Split posts on saved-at divs and count blocks without Instagram URLs."""
    posts = []
    unparsed = 0
    block_start = 0

    for stamp_match in SAVED_AT_RE.finditer(text):
        block = text[block_start:stamp_match.start()]
        post = parse_post_block(block, stamp_match.group(1))
        if post is None:
            unparsed += 1
        else:
            posts.append(post)
        block_start = stamp_match.end()

    return posts, unparsed


def build_manifest(export_dir: Path) -> dict[str, Any]:
    """Parse the two saved-activity export files into one manifest."""
    posts_path = export_dir / "saved_posts.html"
    collections_path = export_dir / "saved_collections.html"
    if not posts_path.is_file():
        raise FileNotFoundError(f"saved posts export not found: {posts_path}")
    if not collections_path.is_file():
        raise FileNotFoundError(f"saved collections export not found: {collections_path}")

    posts_text = posts_path.read_text(encoding="utf-8")
    collections_text = collections_path.read_text(encoding="utf-8")
    collections, shortcode_collections = parse_collections(collections_text)
    posts, unparsed = parse_posts(posts_text)

    matching_names = {
        collection["name"]
        for collection in collections
        if collection["movement_match"]
    }
    for post in posts:
        names = list(shortcode_collections.get(post["shortcode"], []))
        post["collections"] = names
        if any(name in matching_names for name in names):
            post["movement"] = "yes"
        hint_terms = movement_hints(post["caption"], post["hashtags"])
        post["movement_hint"] = bool(hint_terms)
        post["movement_hint_terms"] = hint_terms

    posts.sort(
        key=lambda post: (post["saved_at"] is not None, post["saved_at"] or ""),
        reverse=True,
    )
    movement_yes = sum(post["movement"] == "yes" for post in posts)
    movement_unknown = sum(post["movement"] == "unknown" for post in posts)
    movement_hint = sum(bool(post["movement_hint"]) for post in posts)

    return {
        "_comment": (
            "Instagram saved-post manifest. Movement is yes only when the post "
            "appears in a collection whose name matches an explicit movement keyword."
        ),
        "generated_at": utc_now_iso(),
        "source": {
            "saved_posts": str(posts_path.resolve()),
            "saved_collections": str(collections_path.resolve()),
        },
        "counts": {
            "posts": len(posts),
            "collections": len(collections),
            "movement_yes": movement_yes,
            "movement_unknown": movement_unknown,
            "movement_hint": movement_hint,
            "unparsed_blocks": unparsed,
        },
        "collections": collections,
        "posts": posts,
    }


def command_parse(args: argparse.Namespace) -> int:
    """Implement the parse subcommand."""
    manifest = build_manifest(args.export_dir)
    manifest_path = args.out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    counts = manifest["counts"]
    print(
        f"[ig] wrote {manifest_path}: {counts['posts']} posts, "
        f"{counts['collections']} collections, {counts['movement_yes']} movement=yes, "
        f"{counts['movement_hint']} movement_hint, "
        f"{counts['movement_unknown']} movement=unknown, "
        f"{counts['unparsed_blocks']} unparsed blocks"
    )
    return 0


# ---------------------------------------------------------------- download

def load_download_log(path: Path) -> list[dict[str, Any]]:
    """Load an appendable download log, accepting the legacy object shape."""
    if not path.exists():
        return []
    value = read_json(path)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("attempts"), list):
        return value["attempts"]
    raise ValueError(f"download log must be a list of attempts: {path}")


def download_queue(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order strict movement, hinted movement, then other posts newest-first."""
    newest = lambda post: post.get("saved_at") or ""
    movement = sorted(
        (post for post in posts if post.get("movement") == "yes"),
        key=newest,
        reverse=True,
    )
    hinted = sorted(
        (
            post for post in posts
            if post.get("movement") != "yes" and post.get("movement_hint") is True
        ),
        key=newest,
        reverse=True,
    )
    rest = sorted(
        (
            post for post in posts
            if post.get("movement") != "yes" and post.get("movement_hint") is not True
        ),
        key=newest,
        reverse=True,
    )
    return movement + hinted + rest


def is_auth_failure(error_text: str) -> bool:
    """Classify failures that indicate expired auth or Instagram throttling."""
    lowered = error_text.lower()
    return any(term in lowered for term in AUTH_FAILURE_TERMS)


def run_download(args: argparse.Namespace) -> dict[str, Any]:
    """Run a resumable prioritized download batch and return batch statistics."""
    manifest_path = args.out_dir / "manifest.json"
    manifest = read_json(manifest_path)
    args.clips_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "download_log.json"
    log = load_download_log(log_path)
    done_shortcodes = {
        str(entry.get("shortcode", ""))
        for entry in log
        if entry.get("ok") is True and entry.get("shortcode")
    }
    done_shortcodes.update(
        path.stem for path in completed_files(args.clips_dir)
    )
    queue = [
        post for post in download_queue(manifest.get("posts", []))
        if str(post.get("shortcode", "")) not in done_shortcodes
    ][:args.max_attempts]

    cookies = getattr(args, "cookies", None)
    cookies_from_browser = getattr(args, "cookies_from_browser", None)
    if (
        cookies is None
        and not cookies_from_browser
        and DEFAULT_COOKIES_JAR is not None
        and DEFAULT_COOKIES_JAR.exists()
    ):
        cookies = DEFAULT_COOKIES_JAR
        print(f"[ig] using default cookie jar {DEFAULT_COOKIES_JAR}")

    successes = 0
    attempts = 0
    failed = 0
    consecutive_auth_failures = 0
    auth_aborted = False

    print(
        f"[ig] download queue: {len(queue)} posts, target {args.limit} successes, "
        f"maximum {args.max_attempts} attempts"
    )
    for index, post in enumerate(queue, start=1):
        if successes >= args.limit:
            break
        attempts += 1
        shortcode = str(post.get("shortcode", ""))
        url = str(post.get("url", ""))
        movement = str(post.get("movement", "unknown"))
        started = time.monotonic()

        before = completed_files(args.clips_dir)
        output_template = args.clips_dir / "%(id)s.%(ext)s"
        command = [
            "yt-dlp",
            "-S", "res:720",
            "--no-playlist",
            "--restrict-filenames",
            "-o", str(output_template),
        ]
        if cookies_from_browser:
            command.extend(["--cookies-from-browser", cookies_from_browser])
        if cookies:
            command.extend(["--cookies", str(cookies)])
        command.append(url)

        returncode = None
        error = ""
        diagnostic_text = ""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
            )
            returncode = result.returncode
            error = stderr_sample(result.stderr)
            diagnostic_text = "\n".join((result.stdout or "", result.stderr or ""))
        except subprocess.TimeoutExpired as exc:
            error = stderr_sample(exc.stderr) or "yt-dlp timed out after 300 seconds"
            diagnostic_text = "\n".join(
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes) else (value or "")
                for value in (exc.stdout, exc.stderr)
            )
        except OSError as exc:
            error = str(exc)
            diagnostic_text = error

        after = completed_files(args.clips_dir)
        new_files = sorted(after - before, key=lambda path: path.name.lower())
        matching = [path for path in new_files if path.stem == shortcode]
        downloaded = matching[0] if matching else (new_files[0] if new_files else None)
        ok = returncode == 0 and downloaded is not None
        elapsed = round(time.monotonic() - started, 3)

        entry = {
            "url": url,
            "shortcode": shortcode,
            "movement": movement,
            "ok": ok,
        }
        if ok:
            entry["file"] = relative_to_cwd(downloaded)
            successes += 1
            consecutive_auth_failures = 0
        else:
            failed += 1
            if not error:
                if returncode == 0:
                    error = "yt-dlp returned 0 but created no new file"
                elif returncode is not None:
                    error = f"yt-dlp exited with status {returncode} without stderr"
                else:
                    error = "yt-dlp did not complete"
            entry["error"] = error
            auth_failure = is_auth_failure(diagnostic_text or error)
            if auth_failure:
                entry["auth_failure"] = True
                consecutive_auth_failures += 1
        entry["elapsed_s"] = elapsed
        entry["ts"] = utc_now_iso()
        log.append(entry)
        write_json(log_path, log)

        status = "ok" if ok else "failed"
        print(
            f"[ig] [{index}/{len(queue)}] {shortcode}: {status} "
            f"({successes}/{args.limit} successes, {elapsed:.1f}s)"
        )
        if not ok and error:
            print(f"[ig] error sample: {error}")

        if consecutive_auth_failures >= 3:
            auth_aborted = True
            print(
                "[ig] aborting batch after 3 consecutive auth failures "
                "(cookies likely expired — run refresh-cookies)"
            )
            break

        more_work = index < len(queue) and successes < args.limit
        jitter = getattr(args, "jitter", 4.0)
        delay = random.uniform(args.sleep, args.sleep + jitter)
        if more_work and delay > 0:
            print(f"[ig] sleeping {delay:.1f}s before the next attempt")
            time.sleep(delay)

    print(
        f"[ig] download complete: {successes} successes in {attempts} attempts; "
        f"log {log_path}"
    )
    return {
        "attempted": attempts,
        "ok": successes,
        "failed": failed,
        "auth_aborted": auth_aborted,
    }


def command_download(args: argparse.Namespace) -> int:
    """Implement the prioritized yt-dlp download subcommand."""
    run_download(args)
    return 0


# ---------------------------------------------------------------- nas queue

def command_nas_queue(args: argparse.Namespace) -> int:
    """Write movement-prioritized posts not yet downloaded to the NAS queue."""
    manifest_path = args.out_dir / "manifest.json"
    manifest = read_json(manifest_path)
    log = load_download_log(args.out_dir / "download_log.json")
    downloaded = {
        str(entry.get("shortcode", ""))
        for entry in log
        if isinstance(entry, dict)
        and entry.get("ok") is True
        and entry.get("shortcode")
    }
    posts = [
        post for post in manifest.get("posts", [])
        if isinstance(post, dict)
    ]
    pending = [
        post for post in download_queue(posts)
        if (
            post.get("movement") == "yes"
            or post.get("movement_hint") is True
        )
        and str(post.get("shortcode", "")) not in downloaded
    ]
    records = [
        {
            "url": post.get("url"),
            "shortcode": post.get("shortcode"),
            "saved_at": post.get("saved_at"),
            "movement": post.get("movement", "unknown"),
            "movement_hint_terms": post.get("movement_hint_terms", []),
        }
        for post in pending
    ]
    queue = {
        "_comment": (
            "Movement-prioritized Instagram posts not yet downloaded, for the "
            "standalone Synology NAS ingest runner."
        ),
        "generated_at": utc_now_iso(),
        "count": len(records),
        "posts": records,
    }
    queue_path = args.out_dir / "nas_queue.json"
    write_json(queue_path, queue)
    print(f"[ig] wrote {queue_path}: {len(records)} posts queued for the NAS")
    return 0


# ---------------------------------------------------------------- cookies

def validate_cookie_jar(path: Path | None) -> bool:
    """Return whether a Netscape jar has a usable Instagram session cookie."""
    if path is None or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                if raw_line.startswith("#HttpOnly_"):
                    raw_line = raw_line[len("#HttpOnly_"):]
                elif raw_line.startswith("#"):
                    continue
                if not raw_line.strip():
                    continue
                fields = raw_line.rstrip("\r\n").split("\t")
                if (
                    len(fields) >= 7
                    and "instagram.com" in fields[0].lower()
                    and fields[5].lower() == "sessionid"
                    and len(fields[6]) > 10
                ):
                    return True
    except OSError:
        return False
    return False


def brave_running() -> bool | None:
    """Return Brave process state, or None when tasklist cannot be queried."""
    try:
        result = subprocess.run(
            ["tasklist"], capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return "brave.exe" in (result.stdout or "").lower()


def existing_jar_fallback(reason: str, brave_is_open: bool = False) -> int:
    """Keep a valid prior jar after refresh cannot safely proceed."""
    if validate_cookie_jar(DEFAULT_COOKIES_JAR):
        if brave_is_open:
            print("[ig] Brave is running; kept existing valid jar")
        else:
            print(f"[ig] {reason}; kept existing valid jar")
        return 0
    if brave_is_open:
        print("[ig] Brave is running; close Brave to refresh cookies")
    else:
        print(f"[ig] {reason}; close Brave to refresh cookies")
    return 2


def poll_debugger_websocket(port: int, timeout_s: float = 15.0) -> str | None:
    """Poll Brave's version endpoint until its browser WebSocket is ready."""
    endpoint = f"http://127.0.0.1:{port}/json/version"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                version = json.loads(response.read())
            websocket_url = version.get("webSocketDebuggerUrl")
            if websocket_url:
                return str(websocket_url)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    return None


def ws_read_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly size bytes from a WebSocket connection."""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("WebSocket closed")
        data.extend(chunk)
    return bytes(data)


def ws_send_frame(sock: socket.socket, payload: bytes, opcode: int = 0x1) -> None:
    """Send one final, RFC6455-masked client frame."""
    header = bytearray([0x80 | opcode])
    mask = os.urandom(4)
    size = len(payload)
    if size < 126:
        header.append(0x80 | size)
    elif size < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack(">H", size))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack(">Q", size))
    header.extend(mask)
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    sock.sendall(bytes(header) + masked)


def ws_receive_frame(sock: socket.socket) -> tuple[bool, int, bytes]:
    """Receive one RFC6455 frame from the CDP endpoint."""
    first, second = ws_read_exact(sock, 2)
    final = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    size = second & 0x7F
    if size == 126:
        size = struct.unpack(">H", ws_read_exact(sock, 2))[0]
    elif size == 127:
        size = struct.unpack(">Q", ws_read_exact(sock, 8))[0]
    mask = ws_read_exact(sock, 4) if masked else b""
    payload = ws_read_exact(sock, size)
    if masked:
        payload = bytes(
            byte ^ mask[index % 4] for index, byte in enumerate(payload)
        )
    return final, opcode, payload


def cdp_storage_cookies(websocket_url: str, timeout_s: float = 15.0) -> list[dict[str, Any]]:
    """Call Storage.getCookies over a raw browser-level CDP WebSocket."""
    parsed = urlparse(websocket_url)
    if not parsed.hostname:
        raise ValueError("CDP WebSocket URL has no hostname")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    if parsed.scheme == "wss":
        raise ValueError("local CDP unexpectedly requested a secure WebSocket")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    sock = socket.create_connection((parsed.hostname, port), timeout=10)
    try:
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        sock.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            response.extend(ws_read_exact(sock, 1))
            if len(response) > 65536:
                raise ConnectionError("oversized WebSocket handshake")
        header_text = bytes(response).decode("latin-1", errors="replace")
        if not header_text.startswith("HTTP/1.1 101"):
            raise ConnectionError("CDP WebSocket upgrade was rejected")
        expected_accept = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        if f"sec-websocket-accept: {expected_accept}".lower() not in header_text.lower():
            raise ConnectionError("CDP WebSocket handshake was invalid")

        ws_send_frame(sock, json.dumps(
            {"id": 1, "method": "Storage.getCookies"}
        ).encode("utf-8"))
        deadline = time.monotonic() + timeout_s
        fragments = bytearray()
        text_opcode = None
        while time.monotonic() < deadline:
            sock.settimeout(max(0.1, min(2.0, deadline - time.monotonic())))
            try:
                final, opcode, payload = ws_receive_frame(sock)
            except socket.timeout:
                continue
            if opcode == 0x8:
                raise ConnectionError("CDP WebSocket closed before responding")
            if opcode == 0x9:
                ws_send_frame(sock, payload, opcode=0xA)
                continue
            if opcode in (0x1, 0x2):
                fragments = bytearray(payload)
                text_opcode = opcode
            elif opcode == 0x0 and text_opcode is not None:
                fragments.extend(payload)
            else:
                continue
            if not final:
                continue
            if text_opcode != 0x1:
                fragments.clear()
                text_opcode = None
                continue
            message = json.loads(fragments.decode("utf-8", errors="replace"))
            fragments.clear()
            text_opcode = None
            if message.get("id") == 1:
                if message.get("error"):
                    raise RuntimeError("Storage.getCookies returned a CDP error")
                return list(message["result"]["cookies"])
        raise TimeoutError("Storage.getCookies timed out")
    finally:
        sock.close()


def debug_port_closed(port: int, timeout_s: float = 5.0) -> bool:
    """Wait until the launched Brave process releases its debug port."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            probe = socket.create_connection(("127.0.0.1", port), timeout=0.25)
        except OSError:
            return True
        else:
            probe.close()
        time.sleep(0.25)
    return False


def terminate_launched_brave(process: subprocess.Popen, port: int) -> None:
    """Terminate only this process and verify that its CDP endpoint closes."""
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if not debug_port_closed(port):
        print(f"[ig] warning: launched Brave debug port {port} is still open")


def write_cookie_jar(path: Path, cookies: list[dict[str, Any]]) -> None:
    """Write Instagram CDP cookies in the Netscape cookies.txt format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fallback_expiry = int(time.time()) + 7 * 24 * 60 * 60
    lines = ["# Netscape HTTP Cookie File"]
    for cookie in sorted(
        cookies, key=lambda item: (str(item.get("domain", "")), str(item.get("name", "")))
    ):
        domain = str(cookie.get("domain", ""))
        cookie_path = str(cookie.get("path", "/") or "/")
        secure = "TRUE" if bool(cookie.get("secure")) else "FALSE"
        try:
            expires = int(float(cookie.get("expires", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            expires = 0
        if expires <= 0:
            expires = fallback_expiry
        name = str(cookie.get("name", "")).replace("\t", "")
        value = str(cookie.get("value", "")).replace("\t", "").replace("\r", "").replace("\n", "")
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        lines.append("\t".join((
            domain, include_subdomains, cookie_path, secure, str(expires), name, value,
        )))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def refresh_cookie_jar(debug_port: int = DEFAULT_DEBUG_PORT) -> int:
    """Refresh Instagram cookies by letting real-profile Brave decrypt them."""
    running = brave_running()
    if running is True:
        return existing_jar_fallback("Brave is running", brave_is_open=True)
    if running is None:
        return existing_jar_fallback("could not check whether Brave is running")
    if DEFAULT_COOKIES_JAR is None or REAL_BRAVE_USER_DATA_DIR is None:
        print("[ig] LOCALAPPDATA is unavailable; cannot refresh the cookie jar")
        return 2

    brave_exe = next((path for path in BRAVE_EXE_CANDIDATES if path.is_file()), None)
    if brave_exe is None:
        return existing_jar_fallback("Brave executable was not found")

    creationflags = (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
    command = [
        str(brave_exe),
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={debug_port}",
        f"--user-data-dir={REAL_BRAVE_USER_DATA_DIR}",
        "--profile-directory=Default",
        "--no-first-run",
        "--restore-last-session=false",
        "about:blank",
    ]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except OSError:
        return existing_jar_fallback("could not launch Brave")

    websocket_url = poll_debugger_websocket(debug_port)
    if websocket_url is None:
        terminate_launched_brave(process, debug_port)
        return existing_jar_fallback("Brave debug endpoint did not start")

    refresh_error: Exception | None = None
    instagram_cookies: list[dict[str, Any]] = []
    try:
        cookies = cdp_storage_cookies(websocket_url)
        instagram_cookies = [
            cookie for cookie in cookies
            if "instagram.com" in str(cookie.get("domain", "")).lower()
        ]
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError,
        TimeoutError,
    ) as exc:
        refresh_error = exc
    finally:
        terminate_launched_brave(process, debug_port)
    if refresh_error is not None:
        return existing_jar_fallback(f"cookie refresh failed ({refresh_error})")

    write_cookie_jar(DEFAULT_COOKIES_JAR, instagram_cookies)
    try:
        username = os.environ.get("USERNAME", "")
        subprocess.run(
            [
                "icacls", str(DEFAULT_COOKIES_JAR), "/inheritance:r",
                "/grant:r", f"{username}:F",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass

    if validate_cookie_jar(DEFAULT_COOKIES_JAR):
        print(f"[ig] wrote cookie jar ({len(instagram_cookies)} IG cookies)")
        return 0
    print("[ig] refreshed cookies did not include a valid Instagram session")
    return 2


def command_refresh_cookies(args: argparse.Namespace) -> int:
    """Implement the refresh-cookies subcommand."""
    return refresh_cookie_jar(args.debug_port)


# ---------------------------------------------------------------- sheets

def probe_duration(clip: Path) -> float:
    """Read one clip duration from ffprobe JSON output."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(clip),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(stderr_sample(result.stderr) or f"ffprobe exited {result.returncode}")
    data = json.loads(result.stdout)
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration {duration!r}")
    return duration


def run_ffmpeg(command: list[str], timeout: int = 120) -> None:
    """Run ffmpeg and raise a concise error on failure."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        sample = stderr_sample(exc.stderr)
        raise RuntimeError(sample or f"ffmpeg timed out after {timeout} seconds") from exc
    if result.returncode != 0:
        raise RuntimeError(stderr_sample(result.stderr) or f"ffmpeg exited {result.returncode}")


def existing_sheets(sheet_dir: Path, clip_stem: str) -> list[Path]:
    """Find generated sheet names for one clip without treating stem as a glob."""
    if not sheet_dir.exists():
        return []
    prefix = f"{clip_stem}_dur"
    suffix = "s_sheet.jpg"
    return sorted(
        path for path in sheet_dir.iterdir()
        if path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix)
    )


def make_contact_sheet(clip: Path, sheet_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Probe a clip, extract nine evenly spaced frames, and tile them 3x3."""
    duration = probe_duration(clip)
    frame_times = [duration * (index + 0.5) / 9 for index in range(9)]
    duration_int = int(round(duration))
    sheet_path = sheet_dir / f"{clip.stem}_dur{duration_int}s_sheet.jpg"

    with tempfile.TemporaryDirectory(prefix="ig_sheet_") as temp_name:
        temp_dir = Path(temp_name)
        for index, frame_time in enumerate(frame_times):
            frame_path = temp_dir / f"frame_{index}.jpg"
            run_ffmpeg([
                "ffmpeg", "-y",
                "-ss", f"{frame_time:.6f}",
                "-i", str(clip),
                "-frames:v", "1",
                "-q:v", "3",
                "-vf", "scale=480:-2",
                str(frame_path),
            ])

        temp_sheet = temp_dir / "sheet.jpg"
        run_ffmpeg([
            "ffmpeg", "-y",
            "-pattern_type", "sequence",
            "-start_number", "0",
            "-i", str(temp_dir / "frame_%d.jpg"),
            "-vf", "tile=3x3:padding=4",
            "-frames:v", "1",
            str(temp_sheet),
        ])
        temp_sheet.replace(sheet_path)

    sidecar = {
        "clip": str(clip.resolve()),
        "duration_s": round(duration, 3),
        "frame_times_s": [round(value, 3) for value in frame_times],
        "sheet": str(sheet_path.resolve()),
    }
    return sheet_path, sidecar


def command_sheets(args: argparse.Namespace) -> int:
    """Implement contact-sheet generation for all supported clips."""
    sheet_dir = args.out_dir / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    if not args.clips_dir.exists():
        raise FileNotFoundError(f"clips directory not found: {args.clips_dir}")
    clips = sorted(
        (
            path for path in args.clips_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    made = 0
    skipped = 0
    failed = 0

    print(f"[ig] building contact sheets for {len(clips)} clips")
    for index, clip in enumerate(clips, start=1):
        found = existing_sheets(sheet_dir, clip.stem)
        if found and not args.force:
            skipped += 1
            print(f"[ig] [{index}/{len(clips)}] {clip.name}: existing sheet, skipped")
            continue
        try:
            sheet_path, sidecar = make_contact_sheet(clip, sheet_dir)
            sidecar_path = sheet_dir / f"{clip.stem}_sheet.json"
            write_json(sidecar_path, sidecar)
            made += 1
            print(f"[ig] [{index}/{len(clips)}] {clip.name}: wrote {sheet_path.name}")
        except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            failed += 1
            print(f"[ig] [{index}/{len(clips)}] {clip.name}: failed ({exc})")

    print(f"[ig] sheets complete: {made} made, {skipped} skipped, {failed} failed")
    args._sheet_stats = {"made": made, "skipped": skipped, "failed": failed}
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------- queue

def sidecar_sheet_path(sidecar: dict[str, Any], sheet_dir: Path, shortcode: str) -> Path | None:
    """Resolve a sidecar's sheet path, with a local-directory fallback."""
    value = sidecar.get("sheet")
    if value:
        path = Path(str(value)).expanduser()
        if path.exists():
            return path.resolve()
    found = existing_sheets(sheet_dir, shortcode)
    return found[0].resolve() if found else None


def sidecar_clip_path(sidecar: dict[str, Any], clips_dir: Path, shortcode: str) -> Path:
    """Resolve a sidecar clip path, falling back to the configured clips dir."""
    value = sidecar.get("clip")
    if value:
        return Path(str(value)).expanduser().resolve()
    existing = find_existing_clip(clips_dir, shortcode)
    if existing:
        return existing.resolve()
    return (clips_dir / shortcode).resolve()


def command_queue(args: argparse.Namespace) -> int:
    """Join contact-sheet sidecars to manifest metadata for VLM tagging."""
    manifest_path = args.out_dir / "manifest.json"
    manifest = read_json(manifest_path)
    posts = manifest.get("posts", [])
    posts_by_shortcode = {str(post.get("shortcode")): post for post in posts}
    manifest_order = {
        str(post.get("shortcode")): index for index, post in enumerate(posts)
    }
    sheet_dir = args.out_dir / "sheets"
    sidecars = sorted(sheet_dir.glob("*_sheet.json"), key=lambda path: path.name.lower())
    records = []
    failures = 0

    for sidecar_path in sidecars:
        shortcode = sidecar_path.name[:-len("_sheet.json")]
        try:
            sidecar = read_json(sidecar_path)
            sheet_path = sidecar_sheet_path(sidecar, sheet_dir, shortcode)
            if sheet_path is None:
                raise FileNotFoundError("referenced sheet is missing")
            clip_path = sidecar_clip_path(sidecar, args.clips_dir, shortcode)
            post = posts_by_shortcode.get(shortcode, {})
            caption = str(post.get("caption", "") or "")[:300]
            records.append({
                "shortcode": shortcode,
                "sheet": str(sheet_path.resolve()),
                "clip": str(clip_path.resolve()),
                "duration_s": sidecar.get("duration_s"),
                "frame_times_s": sidecar.get("frame_times_s", []),
                "url": post.get("url"),
                "saved_at": post.get("saved_at"),
                "caption": caption,
                "hashtags": post.get("hashtags", []),
                "owner": post.get("owner", {"username": "", "name": ""}),
                "movement": post.get("movement", "unknown"),
                "movement_hint": bool(post.get("movement_hint", False)),
                "collections": post.get("collections", []),
            })
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            failures += 1
            print(f"[ig] queue skipped {sidecar_path.name}: {exc}")

    records.sort(key=lambda item: (manifest_order.get(item["shortcode"], len(posts)), item["shortcode"]))
    queue = {
        "_comment": "Contact-sheet queue for vision-model movement-library triage.",
        "generated_at": utc_now_iso(),
        "move_vocab": MOVE_VOCAB,
        "tagging_instructions": (
            "Inspect each 3x3 contact sheet; its nine frames are evenly spaced through "
            "the clip and frame_times_s lists their times. For each clip return JSON "
            "{shortcode, moves:[vocabulary names or 'other:<freeform>'], style, "
            "description:<one line>, library_worthiness:'yes'|'maybe'|'no', "
            "worthiness_reason}."
        ),
        "clips": records,
    }
    queue_path = args.out_dir / "tagging_queue.json"
    write_json(queue_path, queue)
    print(f"[ig] wrote {queue_path}: {len(records)} clips, {failures} skipped")
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------- tag

def owner_username(owner: Any) -> str:
    """Return the username from either queue owner representation."""
    if isinstance(owner, dict):
        return str(owner.get("username", ""))
    return str(owner or "")


def is_partner_or_lift(entry: dict[str, Any]) -> bool:
    """Apply the recall-oriented partner/lift policy to one tag entry."""
    moves = entry.get("moves", [])
    if not isinstance(moves, list):
        moves = []
    move_match = any(
        str(move).startswith("partner:") or str(move) in ("lift", "acro")
        for move in moves
    )
    style = str(entry.get("style", "") or "").lower()
    return move_match or any(term in style for term in PARTNER_STYLE_TERMS)


def apply_save_policy(entry: dict[str, Any]) -> None:
    """Keep every clip and protect partner/lift recall for extraction triage."""
    entry["save"] = "yes"
    partner_or_lift = is_partner_or_lift(entry)
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


def tags_document(clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the stable top-level tags.json shape."""
    return {
        "_comment": (
            "Movement-library tags for downloaded Instagram clips. Every clip is "
            "saved permanently; library_worthiness only gates 3D extraction priority."
        ),
        "model": f"{TAG_MODEL} via agy CLI",
        "cost_usd": 0,
        "clips": clips,
    }


def load_and_migrate_tags(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load tags and apply the one-time save/partner schema migration."""
    if path.exists():
        value = read_json(path)
        if not isinstance(value, dict) or not isinstance(value.get("clips"), list):
            raise ValueError(f"tags file must contain a clips list: {path}")
        clips = [dict(entry) for entry in value["clips"] if isinstance(entry, dict)]
    else:
        clips = []
    initial_shortcodes = {
        str(entry.get("shortcode", "")) for entry in clips if entry.get("shortcode")
    }
    for entry in clips:
        apply_save_policy(entry)
    write_json(path, tags_document(clips))
    return clips, initial_shortcodes


def build_tag_prompt(batch: list[dict[str, Any]]) -> str:
    """Build a strict tagging prompt containing absolute contact-sheet paths."""
    clip_summaries = []
    for clip in batch:
        sheet = Path(str(clip.get("sheet", ""))).expanduser().resolve()
        clip_summaries.append({
            "sheet": str(sheet),
            "shortcode": str(clip.get("shortcode", "")),
            "duration_s": clip.get("duration_s"),
            "owner": owner_username(clip.get("owner")),
            "caption": str(clip.get("caption", "") or "")[:120],
        })
    return f"""Inspect each listed 3x3 contact sheet and tag its movement.

Breaking vocabulary (use exactly): toprock, uprock, 6-step, footwork-other, cc, swipe, windmill, flare, 1990, 2000, headspin, backspin, halo, baby-freeze, chair-freeze, airchair, elbow-freeze, handstand-freeze, kick-up, transition, power-combo, other

PARTNER DANCING (Shawn priority): use partner:<style>, for example partner:salsa, partner:bachata, partner:zouk, partner:kizomba, partner:wcs, partner:lindy, partner:tango, partner:ballroom, or partner:social if unsure.

LIFTS/ACRO (Shawn priority): use lift for partner lifts/aerials and acro for acrobatics/acroyoga.

Other movement: use other:<name> (parkour, tricking, tutting, popping, etc.).

For every clip return shortcode, moves[], style, description (one line), library_worthiness ("yes", "maybe", or "no"), and worthiness_reason. library_worthiness means fitness as a clean single-move reference for 3D SMPL/GVHMR extraction, NOT general quality. Return STRICT JSON only in exactly this shape: {{"clips":[...]}}. Preserve the input order.

Clips:
{json.dumps(clip_summaries, ensure_ascii=False, indent=2)}"""


def call_agy_for_tags(batch: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    """Run one agy batch and extract its tail JSON response."""
    result = subprocess.run(
        [
            "agy",
            "-p", build_tag_prompt(batch),
            "--model", TAG_MODEL,
            "--add-dir", str(out_dir.resolve()),
            "--dangerously-skip-permissions",
            "--print-timeout", "20m",
        ],
        capture_output=True,
        text=True,
        timeout=1500,
    )
    if result.returncode != 0:
        raise RuntimeError(
            stderr_sample(result.stderr) or f"agy exited with status {result.returncode}"
        )
    cleaned = ANSI_ESCAPE_RE.sub("", result.stdout or "")
    matches = list(TAG_JSON_RE.finditer(cleaned))
    if not matches:
        raise ValueError("agy output did not contain a clips JSON object")
    response = json.loads(matches[-1].group(0))
    clips = response.get("clips")
    if not isinstance(clips, list):
        raise ValueError("agy clips response was not a list")
    return [dict(entry) for entry in clips if isinstance(entry, dict)]


def absolute_metadata_path(value: Any) -> str:
    """Normalize a required queue path for tags.json metadata joins."""
    if not value:
        return ""
    return str(Path(str(value)).expanduser().resolve())


def merge_tag_batch(
    tagged_clips: list[dict[str, Any]],
    queue_batch: list[dict[str, Any]],
    model_clips: list[dict[str, Any]],
) -> None:
    """Upsert one model response and join authoritative queue metadata."""
    positions = {
        str(entry.get("shortcode", "")): index
        for index, entry in enumerate(tagged_clips)
        if entry.get("shortcode")
    }
    model_by_shortcode = {
        str(entry.get("shortcode", "")): entry
        for entry in model_clips
        if entry.get("shortcode")
    }
    missing = [
        str(clip.get("shortcode", "")) for clip in queue_batch
        if str(clip.get("shortcode", "")) not in model_by_shortcode
    ]
    if missing:
        raise ValueError(f"agy response omitted {len(missing)} requested clips")

    for queue_clip in queue_batch:
        shortcode = str(queue_clip.get("shortcode", ""))
        current = (
            dict(tagged_clips[positions[shortcode]])
            if shortcode in positions else {}
        )
        current.update(model_by_shortcode[shortcode])
        current["shortcode"] = shortcode
        moves = current.get("moves", [])
        current["moves"] = [str(move) for move in moves] if isinstance(moves, list) else []
        current.update({
            "url": queue_clip.get("url"),
            "owner": owner_username(queue_clip.get("owner")),
            "duration_s": queue_clip.get("duration_s"),
            "movement": queue_clip.get("movement", "unknown"),
            "movement_hint": bool(queue_clip.get("movement_hint", False)),
            "collections": queue_clip.get("collections", []),
            "sheet": absolute_metadata_path(queue_clip.get("sheet")),
            "clip": absolute_metadata_path(queue_clip.get("clip")),
        })
        apply_save_policy(current)
        if shortcode in positions:
            tagged_clips[positions[shortcode]] = current
        else:
            positions[shortcode] = len(tagged_clips)
            tagged_clips.append(current)


def summarize_tags(clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Return cumulative counts used by tag output and nightly logs."""
    worthiness = {"yes": 0, "maybe": 0, "no": 0}
    for entry in clips:
        value = str(entry.get("library_worthiness", "")).lower()
        if value in worthiness:
            worthiness[value] += 1
    return {
        "total": len(clips),
        "save_yes": sum(entry.get("save") == "yes" for entry in clips),
        "partner_or_lift": sum(bool(entry.get("partner_or_lift")) for entry in clips),
        "worthiness": worthiness,
    }


def run_tagging(args: argparse.Namespace) -> dict[str, Any]:
    """Tag queued sheets in agy batches and merge each successful response."""
    queue_path = args.out_dir / "tagging_queue.json"
    queue = read_json(queue_path)
    if not isinstance(queue, dict) or not isinstance(queue.get("clips"), list):
        raise ValueError(f"tagging queue must contain a clips list: {queue_path}")
    tags_path = args.out_dir / "tags.json"
    tagged_clips, initial_shortcodes = load_and_migrate_tags(tags_path)
    work = [
        clip for clip in queue["clips"]
        if isinstance(clip, dict)
        and clip.get("shortcode")
        and (args.retag or str(clip.get("shortcode")) not in initial_shortcodes)
    ]

    tagged_this_run = 0
    new_tagged = 0
    failed_batches = 0
    for offset in range(0, len(work), args.batch_size):
        batch = work[offset:offset + args.batch_size]
        batch_number = offset // args.batch_size + 1
        total_batches = math.ceil(len(work) / args.batch_size)
        print(f"[ig] tagging batch {batch_number}/{total_batches}: {len(batch)} clips")
        try:
            model_clips = call_agy_for_tags(batch, args.out_dir)
            merge_tag_batch(tagged_clips, batch, model_clips)
            write_json(tags_path, tags_document(tagged_clips))
            tagged_this_run += len(batch)
            new_tagged += sum(
                str(clip.get("shortcode")) not in initial_shortcodes for clip in batch
            )
        except (
            OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError,
            subprocess.TimeoutExpired,
        ) as exc:
            failed_batches += 1
            print(f"[ig] tagging batch {batch_number} failed: {exc}")

    cumulative = summarize_tags(tagged_clips)
    split = cumulative["worthiness"]
    retagged = tagged_this_run - new_tagged
    retag_text = f", {retagged} retagged" if retagged else ""
    print(
        f"[ig] tag complete: tagged {new_tagged} new{retag_text}; worthiness "
        f"yes={split['yes']}, maybe={split['maybe']}, no={split['no']}; "
        f"save=yes {cumulative['save_yes']}; partner_or_lift "
        f"{cumulative['partner_or_lift']}"
    )
    stats = {
        "tagged_this_run": tagged_this_run,
        "new_tagged": new_tagged,
        "failed_batches": failed_batches,
        "cumulative": cumulative,
    }
    args._tag_stats = stats
    return stats


def command_tag(args: argparse.Namespace) -> int:
    """Implement the tag subcommand."""
    stats = run_tagging(args)
    return 0 if stats["failed_batches"] == 0 else 1


# ---------------------------------------------------------------- extract queue

def joined_tag_text(entry: dict[str, Any], fields: tuple[str, ...]) -> str:
    """Join scalar and list tag fields into one marker-searchable string."""
    parts = []
    for field in fields:
        value = entry.get(field, "")
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value or ""))
    return " ".join(parts)


def extraction_clip_filename(value: Any, shortcode: str) -> str:
    """Return a flat clip filename, accepting Windows or POSIX metadata paths."""
    if not value:
        return f"{shortcode}.mp4"
    name = re.split(r"[\\/]", str(value).rstrip("\\/"))[-1]
    return name or f"{shortcode}.mp4"


def needs_floor_plane_only(entry: dict[str, Any]) -> bool:
    """Flag clips whose movement is better extracted against a floor plane."""
    moves = entry.get("moves", [])
    if not isinstance(moves, list):
        moves = []
    if any(
        str(move).lower().startswith(
            ("other:parkour", "other:tricking", "other:freerun")
        )
        for move in moves
    ):
        return True

    text = joined_tag_text(entry, ("style", "description", "moves"))
    if FLOOR_PLANE_ONLY_RE.search(text):
        return True
    style = str(entry.get("style", "") or "")
    return bool(TRICKING_RE.search(style) and not BREAKING_RE.search(style))


def extraction_queue_sort_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Order clean solo, floor-plane, then review clips deterministically."""
    review = bool(entry.get("review"))
    floor_plane_only = bool(entry.get("floor_plane_only"))
    group = 2 if review else 1 if floor_plane_only else 0
    duration = entry.get("duration_s")
    return (
        group,
        duration is None,
        duration if duration is not None else 0,
        str(entry.get("shortcode", "")),
    )


def command_extract_queue(args: argparse.Namespace) -> int:
    """Build the policy-filtered queue for nightly NAS 3D extraction."""
    tags_path = args.out_dir / "tags.json"
    tags = read_json(tags_path)
    if not isinstance(tags, dict) or not isinstance(tags.get("clips"), list):
        raise ValueError(f"tags file must contain a clips list: {tags_path}")

    tag_entries = tags["clips"]
    candidates = [
        entry for entry in tag_entries
        if isinstance(entry, dict) and entry.get("library_worthiness") == "yes"
    ]
    clips = []
    excluded = []
    for entry in candidates:
        shortcode = str(entry.get("shortcode", ""))
        if entry.get("partner_or_lift"):
            excluded.append({
                "shortcode": shortcode,
                "reason": "partner_or_lift",
                "style": entry.get("style"),
                "description": entry.get("description"),
            })
            continue

        montage_text = joined_tag_text(
            entry,
            ("style", "description", "worthiness_reason", "moves"),
        )
        if MONTAGE_RE.search(montage_text):
            excluded.append({
                "shortcode": shortcode,
                "reason": "montage",
                "style": entry.get("style"),
                "description": entry.get("description"),
            })
            continue

        review_text = joined_tag_text(entry, ("style", "description"))
        clips.append({
            "shortcode": shortcode,
            "clip": extraction_clip_filename(entry.get("clip"), shortcode),
            "duration_s": entry.get("duration_s"),
            "moves": entry.get("moves"),
            "style": entry.get("style"),
            "description": entry.get("description"),
            "review": bool(REVIEW_RE.search(review_text)),
            "floor_plane_only": needs_floor_plane_only(entry),
        })

    clips.sort(key=extraction_queue_sort_key)
    counts = {
        "tags_total": len(tag_entries),
        "yes_total": len(candidates),
        "auto": len(clips),
        "excluded_partner_or_lift": sum(
            entry["reason"] == "partner_or_lift" for entry in excluded
        ),
        "excluded_montage": sum(
            entry["reason"] == "montage" for entry in excluded
        ),
        "review": sum(entry["review"] for entry in clips),
        "floor_plane_only": sum(entry["floor_plane_only"] for entry in clips),
    }
    queue = {
        "_comment": (
            "Policy-filtered Instagram clip queue for nightly NAS GVHMR "
            "3D extraction."
        ),
        "generated_at": utc_now_iso(),
        "policy": {
            "candidate_tier": "library_worthiness=yes",
            "exclude": ["partner_or_lift", "montage"],
            "montage_markers": MONTAGE_MARKERS,
            "review_markers": REVIEW_MARKERS,
            "floor_plane_only_markers": FLOOR_PLANE_ONLY_MARKERS,
        },
        "counts": counts,
        "clips": clips,
        "excluded": excluded,
    }
    queue_path = args.out_dir / "extract_queue.json"
    write_json(queue_path, queue)
    print(
        f"[ig] wrote {queue_path}: tags_total={counts['tags_total']}, "
        f"yes_total={counts['yes_total']}, auto={counts['auto']}; "
        f"excluded partner_or_lift="
        f"{counts['excluded_partner_or_lift']}, montage="
        f"{counts['excluded_montage']}; review={counts['review']}, "
        f"floor_plane_only={counts['floor_plane_only']}"
    )
    return 0


# ---------------------------------------------------------------- nightly

def load_tags_summary(out_dir: Path) -> dict[str, Any]:
    """Load cumulative tags defensively for the nightly run log."""
    path = out_dir / "tags.json"
    try:
        value = read_json(path) if path.exists() else {"clips": []}
        clips = value.get("clips", []) if isinstance(value, dict) else []
        if not isinstance(clips, list):
            clips = []
        return summarize_tags([entry for entry in clips if isinstance(entry, dict)])
    except (OSError, ValueError, json.JSONDecodeError):
        return summarize_tags([])


def remaining_hinted_posts(manifest: dict[str, Any] | None, out_dir: Path) -> int:
    """Count hinted manifest posts not represented by a successful log entry."""
    if not manifest:
        return 0
    hinted = {
        str(post.get("shortcode", ""))
        for post in manifest.get("posts", [])
        if post.get("movement_hint") is True and post.get("shortcode")
    }
    try:
        log = load_download_log(out_dir / "download_log.json")
    except (OSError, ValueError, json.JSONDecodeError):
        log = []
    downloaded = {
        str(entry.get("shortcode", ""))
        for entry in log
        if entry.get("ok") is True and entry.get("shortcode")
    }
    return len(hinted - downloaded)


def append_ingest_log(
    out_dir: Path,
    manifest: dict[str, Any] | None,
    run: dict[str, Any],
) -> None:
    """Append one scheduler-friendly nightly section, creating its help header."""
    log_path = out_dir / "ingest_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        header = """# Instagram Nightly Ingest Log

This file records scheduled Instagram movement-library ingest runs.

- Disable: `schtasks /Change /TN "breakdance-ig-ingest" /DISABLE`
- Enable: `schtasks /Change /TN "breakdance-ig-ingest" /ENABLE`
- Delete: `schtasks /Delete /TN "breakdance-ig-ingest" /F`

"""
        log_path.write_text(header, encoding="utf-8", newline="\n")

    download = run.get("download", {})
    auth_note = " (auth-aborted)" if download.get("auth_aborted") else ""
    cumulative = load_tags_summary(out_dir)
    split = cumulative["worthiness"]
    lines = [
        f"## {run['timestamp']} nightly run",
        "",
        f"- Cookies: {run.get('cookies', 'not checked')}",
        (
            f"- Downloaded: {download.get('ok', 0)} ok, "
            f"{download.get('failed', 0)} failed in "
            f"{download.get('attempted', 0)} attempts{auth_note}"
        ),
        f"- Sheets made: {run.get('sheets_made', 0)}",
        f"- Clips tagged this run: {run.get('tagged_this_run', 0)}",
        (
            f"- Cumulative: {cumulative['total']} clips, "
            f"save=yes {cumulative['save_yes']}, worthiness "
            f"yes={split['yes']}/maybe={split['maybe']}/no={split['no']}, "
            f"partner_or_lift {cumulative['partner_or_lift']}"
        ),
        (
            "- Remaining un-downloaded movement-hinted posts: "
            f"{remaining_hinted_posts(manifest, out_dir)}"
        ),
    ]
    if run.get("note"):
        lines.append(f"- Note: {run['note']}")
    for error in run.get("errors", []):
        lines.append(f"- Stage warning: {error}")
    lines.extend(("", ""))
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def command_nightly(args: argparse.Namespace) -> int:
    """Run the best-effort scheduled ingest orchestration."""
    run: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "cookies": "not checked",
        "download": {"attempted": 0, "ok": 0, "failed": 0, "auth_aborted": False},
        "sheets_made": 0,
        "tagged_this_run": 0,
        "errors": [],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest: dict[str, Any] | None = None

    try:
        if args.reparse or not manifest_path.exists():
            manifest = build_manifest(args.export_dir)
            write_json(manifest_path, manifest)
            counts = manifest["counts"]
            print(
                f"[ig] nightly wrote manifest: {counts['posts']} posts, "
                f"{counts['movement_hint']} movement_hint"
            )
        else:
            manifest = read_json(manifest_path)
            print(f"[ig] nightly loaded {manifest_path}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        run["errors"].append(f"manifest failed: {exc}")
        run["note"] = "fatal manifest failure; remaining stages skipped"
        append_ingest_log(args.out_dir, manifest, run)
        return 1

    if args.skip_cookies:
        if validate_cookie_jar(DEFAULT_COOKIES_JAR):
            run["cookies"] = "refresh skipped; existing jar valid"
        else:
            run["cookies"] = "refresh skipped; no valid jar"
    else:
        try:
            cookie_status = refresh_cookie_jar(DEFAULT_DEBUG_PORT)
        except Exception as exc:  # Scheduler boundary: log and make the safe decision.
            cookie_status = 2
            run["errors"].append(f"cookie refresh failed: {exc}")
        jar_valid = validate_cookie_jar(DEFAULT_COOKIES_JAR)
        run["cookies"] = "valid" if jar_valid else "invalid"
        if cookie_status != 0 and not jar_valid:
            run["note"] = "no valid cookies; skipped download"
            append_ingest_log(args.out_dir, manifest, run)
            print("[ig] no valid cookies; skipped download")
            return 0

    try:
        download_args = argparse.Namespace(
            out_dir=args.out_dir,
            clips_dir=args.clips_dir,
            limit=args.limit,
            max_attempts=max(args.limit * 2, args.limit),
            sleep=args.sleep,
            jitter=args.jitter,
            cookies=(
                DEFAULT_COOKIES_JAR
                if DEFAULT_COOKIES_JAR is not None and DEFAULT_COOKIES_JAR.exists()
                else None
            ),
            cookies_from_browser=None,
        )
        run["download"] = run_download(download_args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        run["errors"].append(f"download failed: {exc}")

    try:
        sheets_args = argparse.Namespace(
            out_dir=args.out_dir,
            clips_dir=args.clips_dir,
            force=False,
        )
        sheet_status = command_sheets(sheets_args)
        run["sheets_made"] = getattr(sheets_args, "_sheet_stats", {}).get("made", 0)
        if sheet_status != 0:
            run["errors"].append("sheets completed with one or more failures")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        run["errors"].append(f"sheets failed: {exc}")

    try:
        queue_args = argparse.Namespace(out_dir=args.out_dir, clips_dir=args.clips_dir)
        queue_status = command_queue(queue_args)
        if queue_status != 0:
            run["errors"].append("queue completed with one or more skipped sidecars")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        run["errors"].append(f"queue failed: {exc}")

    try:
        tag_args = argparse.Namespace(
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            retag=False,
        )
        tag_stats = run_tagging(tag_args)
        run["tagged_this_run"] = tag_stats["tagged_this_run"]
        if tag_stats["failed_batches"]:
            run["errors"].append(
                f"tag failed for {tag_stats['failed_batches']} batch(es)"
            )
    except (
        OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        run["errors"].append(f"tag failed: {exc}")

    append_ingest_log(args.out_dir, manifest, run)
    print(f"[ig] nightly complete; appended {args.out_dir / 'ingest_log.md'}")
    return 0


# ---------------------------------------------------------------- cli

def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add paths accepted by every subcommand."""
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help="directory containing saved_posts.html and saved_collections.html",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="ingest output directory (default: %(default)s)",
    )


def nonnegative_int(value: str) -> int:
    """argparse converter for zero-or-greater integers."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def nonnegative_float(value: str) -> float:
    """argparse converter for zero-or-greater finite floats."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number zero or greater")
    return parsed


def positive_int(value: str) -> int:
    """argparse converter for positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def debug_port(value: str) -> int:
    """argparse converter for a TCP port number."""
    parsed = int(value)
    if parsed < 1 or parsed > 65535:
        raise argparse.ArgumentTypeError("must be between 1 and 65535")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest Instagram saved posts into the movement library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python analyzer_3d/src/ig_ingest.py parse
  python analyzer_3d/src/ig_ingest.py download --limit 12 --max-attempts 24
  python analyzer_3d/src/ig_ingest.py sheets --force
  python analyzer_3d/src/ig_ingest.py queue
  python analyzer_3d/src/ig_ingest.py nas-queue
  python analyzer_3d/src/ig_ingest.py refresh-cookies
  python analyzer_3d/src/ig_ingest.py tag --batch-size 12
  python analyzer_3d/src/ig_ingest.py extract-queue
  python analyzer_3d/src/ig_ingest.py nightly --limit 45
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="parse exports into manifest.json")
    add_common_arguments(parse_parser)
    parse_parser.set_defaults(func=command_parse)

    download_parser = subparsers.add_parser("download", help="download prioritized posts")
    add_common_arguments(download_parser)
    download_parser.add_argument(
        "--limit", type=nonnegative_int, default=12,
        help="stop after this many successful downloads (default: %(default)s)",
    )
    download_parser.add_argument(
        "--max-attempts", type=nonnegative_int, default=24,
        help="maximum queue entries to attempt (default: %(default)s)",
    )
    download_parser.add_argument(
        "--sleep", type=nonnegative_float, default=8.0,
        help="minimum seconds between network attempts (default: %(default)s)",
    )
    download_parser.add_argument(
        "--jitter", type=nonnegative_float, default=4.0,
        help="random seconds added to --sleep (default: %(default)s)",
    )
    download_parser.add_argument(
        "--cookies-from-browser", default=None, metavar="BROWSER",
        help="pass a browser name to yt-dlp --cookies-from-browser",
    )
    download_parser.add_argument(
        "--cookies", type=Path, default=None, metavar="FILE",
        help="pass a Netscape cookies.txt to yt-dlp --cookies "
             "(use for app-bound-encrypted browsers where --cookies-from-browser fails)",
    )
    download_parser.add_argument(
        "--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
        help="download destination (default: %(default)s)",
    )
    download_parser.set_defaults(func=command_download)

    sheets_parser = subparsers.add_parser("sheets", help="build 3x3 contact sheets")
    add_common_arguments(sheets_parser)
    sheets_parser.add_argument(
        "--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
        help="directory containing downloaded clips (default: %(default)s)",
    )
    sheets_parser.add_argument(
        "--force", action="store_true",
        help="replace contact sheets even when they already exist",
    )
    sheets_parser.set_defaults(func=command_sheets)

    queue_parser = subparsers.add_parser("queue", help="build VLM tagging_queue.json")
    add_common_arguments(queue_parser)
    queue_parser.add_argument(
        "--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
        help="fallback directory for resolving clip paths (default: %(default)s)",
    )
    queue_parser.set_defaults(func=command_queue)

    nas_queue_parser = subparsers.add_parser(
        "nas-queue", help="build movement-prioritized nas_queue.json",
    )
    add_common_arguments(nas_queue_parser)
    nas_queue_parser.set_defaults(func=command_nas_queue)

    refresh_parser = subparsers.add_parser(
        "refresh-cookies",
        help="refresh the external Instagram cookie jar through headless Brave",
    )
    refresh_parser.add_argument(
        "--debug-port", type=debug_port, default=DEFAULT_DEBUG_PORT,
        help="Brave CDP debug port (default: %(default)s)",
    )
    refresh_parser.set_defaults(func=command_refresh_cookies)

    tag_parser = subparsers.add_parser(
        "tag", help="tag queued contact sheets through the agy CLI",
    )
    add_common_arguments(tag_parser)
    tag_parser.add_argument(
        "--batch-size", type=positive_int, default=12,
        help="clips sent in each agy call (default: %(default)s)",
    )
    tag_parser.add_argument(
        "--retag", action="store_true",
        help="tag every queued clip even when it already exists in tags.json",
    )
    tag_parser.set_defaults(func=command_tag)

    extract_queue_parser = subparsers.add_parser(
        "extract-queue", help="build policy-filtered extract_queue.json",
    )
    add_common_arguments(extract_queue_parser)
    extract_queue_parser.set_defaults(func=command_extract_queue)

    nightly_parser = subparsers.add_parser(
        "nightly", help="run the resilient scheduled ingest pipeline",
    )
    add_common_arguments(nightly_parser)
    nightly_parser.add_argument(
        "--limit", type=nonnegative_int, default=45,
        help="target new successful downloads (default: %(default)s)",
    )
    nightly_parser.add_argument(
        "--sleep", type=nonnegative_float, default=8.0,
        help="minimum seconds between network attempts (default: %(default)s)",
    )
    nightly_parser.add_argument(
        "--jitter", type=nonnegative_float, default=4.0,
        help="random seconds added to --sleep (default: %(default)s)",
    )
    nightly_parser.add_argument(
        "--batch-size", type=positive_int, default=12,
        help="clips sent in each agy call (default: %(default)s)",
    )
    nightly_parser.add_argument(
        "--skip-cookies", action="store_true",
        help="skip the Brave cookie refresh stage",
    )
    nightly_parser.add_argument(
        "--reparse", action="store_true",
        help="rebuild manifest.json even when it already exists",
    )
    nightly_parser.add_argument(
        "--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR,
        help="download destination and sheet source (default: %(default)s)",
    )
    nightly_parser.set_defaults(func=command_nightly)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the selected ingest subcommand."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    args = parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("[ig] interrupted")
        return 130
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ig] error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
