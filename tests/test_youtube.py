import pytest

from core.youtube import normalize_entry, parse_playlist_info, uploads_playlist_url


class TestUploadsPlaylistUrl:
    def test_converts_channel_id(self):
        assert (
            uploads_playlist_url("UC-PaZZpjgJ61wkK9yKfpe8w")
            == "https://www.youtube.com/playlist?list=UU-PaZZpjgJ61wkK9yKfpe8w"
        )

    def test_rejects_non_channel_id(self):
        with pytest.raises(ValueError):
            uploads_playlist_url("PL1234567890")


class TestParsePlaylistInfo:
    def test_parses_mocked_ytdlp_dump(self):
        info = {
            "entries": [
                {
                    "id": "mtauI51tE6w",
                    "title": "How to Make a Killer YouTube Intro",
                    "upload_date": "20250401",
                    "view_count": 12345,
                    "duration": 900,
                    "webpage_url": "https://www.youtube.com/watch?v=mtauI51tE6w",
                    "uploader": "Aprilynne Alter",
                },
                None,  # yt-dlp emits None for failed extractions with ignoreerrors
                {"id": "bad", "title": "invalid id length"},
            ]
        }
        videos = parse_playlist_info(info)
        assert len(videos) == 1
        assert videos[0]["video_id"] == "mtauI51tE6w"
        assert videos[0]["channel_name"] == "Aprilynne Alter"

    def test_empty_or_none_info(self):
        assert parse_playlist_info(None) == []
        assert parse_playlist_info({}) == []

    def test_normalize_entry_fills_url_fallback(self):
        video = normalize_entry({"id": "abcdefghijk"})
        assert video["video_url"] == "https://www.youtube.com/watch?v=abcdefghijk"
        assert video["title"] == ""


class TestFetchTranscript:
    def test_returns_none_when_transcript_unavailable(self, monkeypatch):
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import TranscriptsDisabled

        def raise_disabled(self, video_id, languages=None):
            raise TranscriptsDisabled(video_id)

        monkeypatch.setattr(YouTubeTranscriptApi, "fetch", raise_disabled)
        from core.youtube import fetch_transcript

        assert fetch_transcript("mtauI51tE6w") is None

    def test_rate_limit_raises_blocked_not_none(self, monkeypatch):
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import IpBlocked

        def raise_blocked(self, video_id, languages=None):
            raise IpBlocked(video_id)

        monkeypatch.setattr(YouTubeTranscriptApi, "fetch", raise_blocked)
        from core.youtube import TranscriptFetchBlocked, fetch_transcript

        with pytest.raises(TranscriptFetchBlocked):
            fetch_transcript("mtauI51tE6w")

    def test_joins_snippets(self, monkeypatch):
        from youtube_transcript_api import YouTubeTranscriptApi

        class Snippet:
            def __init__(self, text):
                self.text = text

        monkeypatch.setattr(
            YouTubeTranscriptApi,
            "fetch",
            lambda self, video_id, languages=None: [Snippet("hello"), Snippet("world")],
        )
        from core.youtube import fetch_transcript

        assert fetch_transcript("mtauI51tE6w") == "hello world"
