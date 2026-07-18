"""Tests for hybrid (BM25 + dense) retrieval, RRF fusion, and reranking.

All offline: the FakeEmbedder (conftest) stands in for MiniLM and a tiny
FakeCrossEncoder stands in for the sentence-transformers CrossEncoder, so no
model downloads happen.
"""

import pytest

from core.pipeline import build_faiss_index, chunk_corpus, embed_texts, write_index_pair
from core.retrieval import (
    bm25_candidates,
    build_bm25,
    load_index_pair,
    reciprocal_rank_fusion,
    retrieve,
    search,
    tokenize,
)


class FakeCrossEncoder:
    """Scores a (query, text) pair 1.0 if `keyword` is in the text, else 0.0."""

    def __init__(self, keyword):
        self.keyword = keyword

    def predict(self, pairs):
        return [1.0 if self.keyword in text.lower() else 0.0 for _q, text in pairs]


def _video(vid):
    return {
        "video_id": vid, "title": vid, "publish_date": "20250101", "view_count": 1,
        "duration": 60, "video_url": f"https://youtu.be/{vid}", "channel_name": "Test",
    }


# Several topically-distinct videos so each keyword is rare across the corpus
# (BM25 gives common terms near-zero IDF, which is realistic but useless for a
# unit test on a toy corpus — this mirrors the real 1141-chunk distribution).
VARIED_VIDEOS = [_video(v) for v in ("vtitle", "vthumb", "vedit", "videa", "vgrow")]
VARIED_TRANSCRIPTS = {
    "vtitle": "Write better video titles using strong power words for your channel. " * 20,
    "vthumb": "Designing a clickable thumbnail with bold contrast and clear faces. " * 20,
    "vedit": "Editing your footage and pacing your cuts to keep retention high. " * 20,
    "videa": "Coming up with a killer video idea that hooks the whole audience. " * 20,
    "vgrow": "Growing your subscriber base with consistent uploads and daily habits. " * 20,
}


def build_pair(tmp_path, fake_embedder, videos=VARIED_VIDEOS, transcripts=VARIED_TRANSCRIPTS):
    chunks = chunk_corpus(videos, transcripts, chunk_size=300, chunk_overlap=50)
    embeddings = embed_texts(fake_embedder, [c["chunk_text"] for c in chunks])
    index = build_faiss_index(embeddings)
    ip, mp = tmp_path / "i.index", tmp_path / "m.json"
    write_index_pair(index, chunks, ip, mp, {"channel_id": "UCtest"})
    index, metadata = load_index_pair(ip, mp)
    return index, metadata["chunks"]


class TestTokenize:
    def test_lowercases_and_splits_alphanumeric(self):
        assert tokenize("Titles, Thumbnails & INTROS!") == ["titles", "thumbnails", "intros"]

    def test_handles_none_and_empty(self):
        assert tokenize(None) == []
        assert tokenize("") == []


class TestReciprocalRankFusion:
    def test_item_high_in_both_lists_wins(self):
        # id 7 is rank 1 in the first list and rank 2 in the second -> top.
        fused = reciprocal_rank_fusion([[7, 1, 2], [3, 7, 4]], k=60)
        assert fused[0] == 7

    def test_union_of_all_ids_preserved(self):
        fused = reciprocal_rank_fusion([[1, 2], [2, 3]], k=60)
        assert set(fused) == {1, 2, 3}

    def test_empty_lists_give_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestBM25:
    def test_finds_chunk_by_exact_term(self, tmp_path, fake_embedder):
        _, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        hits = bm25_candidates(bm25, "clickable thumbnail bold contrast", k=5)
        assert hits
        assert any("thumbnail" in chunks[i]["chunk_text"].lower() for i in hits)

    def test_zero_overlap_query_returns_nothing(self, tmp_path, fake_embedder):
        _, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        assert bm25_candidates(bm25, "quantum chromodynamics", k=5) == []


class TestRetrieveModes:
    def test_dense_only_matches_search(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        via_retrieve = retrieve(index, chunks, fake_embedder, "video titles", top_k=3)
        via_search = search(index, chunks, fake_embedder.encode(["video titles"]), top_k=3)
        assert [r["chunk_text"] for r in via_retrieve] == [r["chunk_text"] for r in via_search]

    def test_hybrid_surfaces_keyword_match_dense_misses(self, tmp_path, fake_embedder):
        # FakeEmbedder vectors are ~random, so dense alone won't reliably rank
        # the thumbnail chunk first; BM25 fusion must pull it in.
        index, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        results = retrieve(
            index, chunks, fake_embedder, "clickable thumbnail bold contrast",
            top_k=3, hybrid=True, bm25=bm25,
        )
        assert any("thumbnail" in r["chunk_text"].lower() for r in results)

    def test_rerank_reorders_pool_by_cross_encoder(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        results = retrieve(
            index, chunks, fake_embedder, "clickable thumbnail contrast", top_k=1,
            hybrid=True, rerank=True, bm25=bm25,
            cross_encoder=FakeCrossEncoder("thumbnail"),
        )
        assert results
        assert "thumbnail" in results[0]["chunk_text"].lower()

    def test_results_carry_dense_distance_score(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        results = retrieve(index, chunks, fake_embedder, "video titles", top_k=2, hybrid=True, bm25=bm25)
        assert all(isinstance(r["score"], float) and r["score"] >= 0 for r in results)

    def test_threshold_gate_filters_offtopic(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        # Off-topic (no term overlap) + tight distance gate -> nothing survives.
        results = retrieve(
            index, chunks, fake_embedder, "quantum chromodynamics lagrangian",
            top_k=5, hybrid=True, bm25=bm25, max_distance=0.5,
        )
        assert results == []

    def test_hybrid_without_bm25_raises(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        with pytest.raises(ValueError, match="bm25"):
            retrieve(index, chunks, fake_embedder, "x", hybrid=True)

    def test_rerank_without_cross_encoder_raises(self, tmp_path, fake_embedder):
        index, chunks = build_pair(tmp_path, fake_embedder)
        bm25 = build_bm25(chunks)
        with pytest.raises(ValueError, match="cross_encoder"):
            retrieve(index, chunks, fake_embedder, "x", hybrid=True, bm25=bm25, rerank=True)
