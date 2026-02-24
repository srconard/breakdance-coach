"""YouTube video downloader using yt-dlp."""

import os
import re
from pathlib import Path
from typing import Optional

import yt_dlp

# Add deno to PATH if not already there
DENO_DIR = r"C:\Users\Shawn\AppData\Local\Microsoft\WinGet\Packages\DenoLand.Deno_Microsoft.Winget.Source_8wekyb3d8bbwe"
if DENO_DIR not in os.environ.get('PATH', ''):
    os.environ['PATH'] = DENO_DIR + os.pathsep + os.environ.get('PATH', '')


def extract_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL.

    Args:
        url: YouTube URL in various formats

    Returns:
        The video ID string

    Raises:
        ValueError: If the URL is not a valid YouTube URL
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract video ID from URL: {url}")


def download_video(
    url: str,
    output_dir: str = ".",
    filename: Optional[str] = None,
    quality: str = "best_mp4",
) -> tuple[Path, str]:
    """Download a YouTube video with audio.

    Args:
        url: YouTube video URL
        output_dir: Directory to save the video
        filename: Optional custom filename (without extension)
        quality: Quality preset:
            - "best_mp4": Best quality with audio in mp4 (default, fast)
            - "720p": 720p video + best audio, merged
            - "1080p": 1080p video + best audio, merged
            - "best": Absolute best quality video + audio, merged

    Returns:
        Tuple of (path to downloaded video, video title)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # First, get video info to retrieve the title
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info.get('title', 'video')

    # Sanitize filename
    if filename is None:
        filename = re.sub(r'[<>:"/\\|?*]', '_', video_title)

    # Quality presets
    format_map = {
        "best_mp4": "best[ext=mp4]/best",
        "720p": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
        "1080p": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
        "best": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
    }
    format_str = format_map.get(quality, quality)  # Allow raw format strings too

    output_template = str(output_path / f"{filename}.%(ext)s")

    ydl_opts = {
        'format': format_str,
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [_progress_hook],
        # Merge format for separate video+audio streams
        'merge_output_format': 'mp4',
        # Retry options
        'retries': 30,
        'fragment_retries': 30,
        'file_access_retries': 15,
        # Network options - smaller chunks to avoid throttling
        'socket_timeout': 60,
        'http_chunk_size': 1048576,  # 1MB chunks (smaller to avoid throttling)
        # Sleep between retries and requests
        'sleep_interval_requests': 2,
        'sleep_interval': 3,
        # Use android client which often works better
        'extractor_args': {'youtube': {'player_client': ['android_vr', 'tv']}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file
    downloaded_files = list(output_path.glob(f"{filename}.*"))
    if not downloaded_files:
        raise FileNotFoundError(f"Downloaded video not found in {output_path}")

    video_path = downloaded_files[0]
    print(f"\nDownloaded: {video_path}")

    return video_path, video_title


def _progress_hook(d: dict) -> None:
    """Progress hook for yt-dlp downloads."""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        print(f"\rDownloading: {percent} at {speed}", end='', flush=True)
    elif d['status'] == 'finished':
        print("\nDownload complete, processing...")


if __name__ == "__main__":
    # Test the downloader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m shared.downloader <youtube_url>")
        sys.exit(1)

    video_path, title = download_video(sys.argv[1], output_dir="downloads")
    print(f"Downloaded '{title}' to {video_path}")
