import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Deterministic stand-in for SentenceTransformer (no torch needed).

    Maps each text to a 8-dim unit vector derived from a stable hash, so
    identical texts always land on identical vectors.
    """

    dim = 8

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=self.dim)
            vectors.append(v / np.linalg.norm(v))
        return np.asarray(vectors, dtype="float32")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fixture_videos():
    return [
        {
            "video_id": "aaaaaaaaaaa",
            "title": "How to title videos",
            "publish_date": "20250101",
            "view_count": 1000,
            "duration": 600,
            "video_url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
            "channel_name": "Test Channel",
        },
        {
            "video_id": "bbbbbbbbbbb",
            "title": "Thumbnail design basics",
            "publish_date": "20250201",
            "view_count": 2000,
            "duration": 300,
            "video_url": "https://www.youtube.com/watch?v=bbbbbbbbbbb",
            "channel_name": "Test Channel",
        },
        {
            "video_id": "ccccccccccc",
            "title": "No transcript here",
            "publish_date": "20250301",
            "view_count": 10,
            "duration": 60,
            "video_url": "https://www.youtube.com/watch?v=ccccccccccc",
            "channel_name": "Test Channel",
        },
    ]


@pytest.fixture
def fixture_transcripts():
    return {
        "aaaaaaaaaaa": "Great titles win clicks. " * 120,
        "bbbbbbbbbbb": "Thumbnails should be simple and bold. " * 80,
        "ccccccccccc": "",
    }
