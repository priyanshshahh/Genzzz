import json

import numpy as np
import pytest

from core.pipeline import build_faiss_index, chunk_corpus, embed_texts, write_index_pair
from core.retrieval import IndexMetadataMismatch, load_index_pair, search


def build_tiny_pair(tmp_path, fake_embedder, videos, transcripts):
    chunks = chunk_corpus(videos, transcripts, chunk_size=300, chunk_overlap=50)
    embeddings = embed_texts(fake_embedder, [c["chunk_text"] for c in chunks])
    index = build_faiss_index(embeddings)
    index_path = tmp_path / "faiss_index.index"
    metadata_path = tmp_path / "faiss_metadata.json"
    write_index_pair(index, chunks, index_path, metadata_path, {"channel_id": "UCtest"})
    return chunks, embeddings, index_path, metadata_path


class TestIndexBuild:
    def test_build_writes_matched_pair(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        chunks, _, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        assert index.ntotal == len(chunks) == len(metadata["chunks"])
        assert metadata["build_info"]["index_dim"] == fake_embedder.dim
        assert metadata["build_info"]["channel_id"] == "UCtest"

    def test_write_refuses_mismatched_pair(self, tmp_path, fake_embedder):
        embeddings = fake_embedder.encode(["one", "two", "three"])
        index = build_faiss_index(embeddings)
        with pytest.raises(ValueError, match="mismatched"):
            write_index_pair(index, [{"chunk_text": "one"}], tmp_path / "i.index", tmp_path / "m.json")

    def test_build_rejects_empty_embeddings(self):
        with pytest.raises(ValueError):
            build_faiss_index(np.empty((0, 8), dtype="float32"))


class TestPairingInvariant:
    def test_missing_files_raise_with_build_instructions(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_index.py"):
            load_index_pair(tmp_path / "nope.index", tmp_path / "nope.json")

    def test_tampered_metadata_count_raises(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, _, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        payload = json.loads(metadata_path.read_text())
        payload["chunks"] = payload["chunks"][:-1]  # desync
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(IndexMetadataMismatch, match="out of sync"):
            load_index_pair(index_path, metadata_path)

    def test_wrong_declared_dim_raises(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, _, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        payload = json.loads(metadata_path.read_text())
        payload["build_info"]["index_dim"] = 999
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(IndexMetadataMismatch, match="dim"):
            load_index_pair(index_path, metadata_path)

    def test_unsupported_schema_version_raises(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, _, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        payload = json.loads(metadata_path.read_text())
        payload["schema_version"] = 999
        metadata_path.write_text(json.dumps(payload))
        with pytest.raises(IndexMetadataMismatch, match="schema_version"):
            load_index_pair(index_path, metadata_path)


class TestThresholdedSearch:
    def test_exact_match_returned_first(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        chunks, embeddings, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        results = search(index, metadata["chunks"], embeddings[0], top_k=3, max_distance=4.0)
        assert results
        assert results[0]["chunk_text"] == chunks[0]["chunk_text"]
        assert results[0]["score"] == pytest.approx(0.0, abs=1e-5)

    def test_threshold_filters_far_results(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        chunks, embeddings, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        # A query identical to a stored chunk: distance 0 passes, random
        # unit vectors sit near d^2 ~= 2 and must be filtered out.
        results = search(index, metadata["chunks"], embeddings[0], top_k=5, max_distance=0.5)
        assert len(results) == 1

    def test_offtopic_query_returns_empty(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, _, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        offtopic = fake_embedder.encode(["completely unrelated query about quantum gravity"])[0]
        assert search(index, metadata["chunks"], offtopic, top_k=5, max_distance=0.5) == []

    def test_no_threshold_returns_top_k(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, embeddings, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        results = search(index, metadata["chunks"], embeddings[0], top_k=4, max_distance=None)
        assert len(results) == 4

    def test_scores_sorted_ascending(self, tmp_path, fake_embedder, fixture_videos, fixture_transcripts):
        _, embeddings, index_path, metadata_path = build_tiny_pair(
            tmp_path, fake_embedder, fixture_videos, fixture_transcripts
        )
        index, metadata = load_index_pair(index_path, metadata_path)
        results = search(index, metadata["chunks"], embeddings[0], top_k=5, max_distance=None)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores)
