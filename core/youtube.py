"""YouTube scraping helpers: channel video listing + transcript fetching.

Network-touching functions keep their heavy imports (yt_dlp,
youtube_transcript_api, whisper) inside the function body so the offline
test-suite and the Streamlit app never pay for them.
"""

from __future__ import annotations

VIDEO_ID_LEN = 11

# Lean per-video metadata fields carried into every chunk record.
VIDEO_FIELDS = (
    "video_id",
    "title",
    "publish_date",
    "view_count",
    "duration",
    "video_url",
    "channel_name",
)


def uploads_playlist_url(channel_id: str) -> str:
    """Convert a UC... channel id into its uploads-playlist URL."""
    if not channel_id.startswith("UC"):
        raise ValueError(f"Expected a channel id starting with 'UC', got {channel_id!r}")
    return f"https://www.youtube.com/playlist?list=UU{channel_id[2:]}"


def normalize_entry(entry: dict) -> dict:
    """Reduce a yt-dlp playlist entry to the lean schema used downstream."""
    video_id = entry.get("id")
    return {
        "video_id": video_id,
        "title": entry.get("title") or "",
        "publish_date": entry.get("upload_date"),
        "view_count": entry.get("view_count"),
        "duration": entry.get("duration"),
        "video_url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "channel_name": entry.get("uploader") or entry.get("channel") or "",
    }


def parse_playlist_info(info: dict | None) -> list[dict]:
    """Extract valid, normalized video records from a yt-dlp playlist dump."""
    videos = []
    for entry in (info or {}).get("entries") or []:
        if not entry:
            continue
        vid = entry.get("id")
        if not isinstance(vid, str) or len(vid) != VIDEO_ID_LEN:
            continue
        videos.append(normalize_entry(entry))
    return videos


def fetch_channel_videos(channel_id: str) -> list[dict]:
    """Fetch metadata for every upload on a channel (single playlist call)."""
    import yt_dlp

    opts = {
        "skip_download": True,
        "extract_flat": False,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(uploads_playlist_url(channel_id), download=False)
    return parse_playlist_info(info)


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> str | None:
    """Fetch a caption transcript as plain text; None if unavailable."""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import CouldNotRetrieveTranscript

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    except CouldNotRetrieveTranscript:
        return None
    return " ".join(snippet.text for snippet in fetched)


def transcribe_with_whisper(video_id: str, model_name: str = "small.en") -> str | None:
    """Fallback: download audio with yt-dlp and transcribe with Whisper.

    Optional and slow — only used when scripts/build_index.py is run with
    --whisper. Requires `openai-whisper` and ffmpeg to be installed.
    """
    import os

    import yt_dlp

    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Whisper fallback requested but `openai-whisper` is not installed. "
            "Run: pip install openai-whisper (ffmpeg required)."
        ) from exc

    audio_path = f"{video_id}.m4a"
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": audio_path,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtu.be/{video_id}"])
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception:
        return None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
