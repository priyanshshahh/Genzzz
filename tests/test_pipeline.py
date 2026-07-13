from core.pipeline import chunk_corpus, chunk_transcript, clean_text
from core.youtube import VIDEO_FIELDS


class TestCleanText:
    def test_removes_timestamps(self):
        assert clean_text("intro [0:12] middle 1:02:33 end") == "intro middle end"

    def test_removes_urls(self):
        assert clean_text("see https://example.com/x?y=1 for more") == "see for more"

    def test_collapses_whitespace_and_newlines(self):
        assert clean_text("a\n\nb\tc   d\r") == "a b c d"

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""
        assert clean_text(42) == ""


class TestChunking:
    def test_chunks_respect_size_limit(self):
        text = "word " * 1000
        chunks = chunk_transcript(text.strip(), chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_overlap_carries_shared_text(self):
        text = ". ".join(f"sentence {i}" for i in range(200))
        chunks = chunk_transcript(text, chunk_size=300, chunk_overlap=100)
        assert len(chunks) >= 2
        # consecutive chunks share at least one word due to overlap
        assert set(chunks[0].split()) & set(chunks[1].split())

    def test_short_text_single_chunk(self):
        assert chunk_transcript("tiny text", 1000, 200) == ["tiny text"]


class TestChunkCorpus:
    def test_skips_videos_without_transcript(self, fixture_videos, fixture_transcripts):
        records = chunk_corpus(fixture_videos, fixture_transcripts)
        assert records
        assert all(r["video_id"] != "ccccccccccc" for r in records)

    def test_records_carry_metadata_and_chunk_fields(self, fixture_videos, fixture_transcripts):
        records = chunk_corpus(fixture_videos, fixture_transcripts)
        for record in records:
            for field in VIDEO_FIELDS:
                assert field in record
            assert "chunk_index" in record
            assert record["chunk_text"].strip()

    def test_chunk_indices_are_sequential_per_video(self, fixture_videos, fixture_transcripts):
        records = chunk_corpus(fixture_videos, fixture_transcripts)
        by_video = {}
        for r in records:
            by_video.setdefault(r["video_id"], []).append(r["chunk_index"])
        for indices in by_video.values():
            assert indices == list(range(len(indices)))
