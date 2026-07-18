"""Loading the index/metadata pair and thresholded similarity search."""

from __future__ import annotations

import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class IndexMetadataMismatch(RuntimeError):
    """The FAISS index and metadata JSON do not describe the same build."""


# The metadata schema this reader understands. Kept in sync with
# core.pipeline.METADATA_SCHEMA_VERSION (the writer). Bump both together
# whenever the chunk-record shape changes incompatibly.
SUPPORTED_SCHEMA_VERSION = 1


def load_index_pair(index_path: Path, metadata_path: Path):
    """Load and validate the FAISS index + metadata pair.

    Raises FileNotFoundError if either half is missing and
    IndexMetadataMismatch if vector count or dimension disagree.
    Returns (index, metadata_dict).
    """
    import faiss

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    missing = [str(p) for p in (index_path, metadata_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing index artifacts: "
            + ", ".join(missing)
            + ". Build them with: python scripts/build_index.py"
        )

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    schema_version = metadata.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise IndexMetadataMismatch(
            f"{metadata_path} has schema_version {schema_version!r}, but this "
            f"code understands version {SUPPORTED_SCHEMA_VERSION}. Rebuild the "
            "index with scripts/build_index.py after upgrading."
        )

    chunks = metadata.get("chunks")
    if not isinstance(chunks, list):
        raise IndexMetadataMismatch(
            f"{metadata_path} has no 'chunks' list — regenerate with scripts/build_index.py"
        )
    if index.ntotal != len(chunks):
        raise IndexMetadataMismatch(
            f"Index has {index.ntotal} vectors but metadata has {len(chunks)} chunks "
            "— the pair is out of sync; rebuild with scripts/build_index.py"
        )
    declared_dim = (metadata.get("build_info") or {}).get("index_dim")
    if declared_dim is not None and declared_dim != index.d:
        raise IndexMetadataMismatch(
            f"Metadata declares dim={declared_dim} but index dim={index.d}"
        )
    return index, metadata


def search(
    index,
    chunks: list[dict],
    query_embedding,
    top_k: int = 4,
    max_distance: float | None = None,
) -> list[dict]:
    """Search the index; return chunk records with a `score` (squared L2).

    Results with distance above `max_distance` are dropped, so off-topic
    queries yield an empty list instead of forced nearest neighbours.
    """
    import numpy as np

    query = np.asarray(query_embedding, dtype="float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances, indices = index.search(query, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < 0:
            continue
        if max_distance is not None and dist > max_distance:
            continue
        record = dict(chunks[idx])
        record["score"] = float(dist)
        results.append(record)
    return results


def embed_query(embedder, question: str):
    """Embed a single query string as float32."""
    import numpy as np

    return np.asarray(embedder.encode([question]), dtype="float32")


# --- Hybrid retrieval (BM25 + dense) + cross-encoder reranking -------------
#
# Dense-only `search()` above is the honest baseline. The functions below add
# the two production-standard first-/second-stage improvements the benchmark
# calls for, gated by config flags so they can be A/B'd on the same index:
#   * BM25 sparse retrieval fused with dense via reciprocal-rank fusion, so
#     exact-term/proper-noun queries the dense model ranks too low still
#     surface into the candidate pool.
#   * A CPU cross-encoder that reorders the fused pool (only helps once the
#     target is in the pool, hence it runs after fusion).


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by BM25 build and query time."""
    return _TOKEN_RE.findall((text or "").lower())


def build_bm25(chunks: list[dict]):
    """Build a BM25Okapi index over the chunks' `chunk_text` (in-memory)."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(c.get("chunk_text", "")) for c in chunks])


def bm25_candidates(bm25, query: str, k: int) -> list[int]:
    """Return up to `k` chunk indices ranked by BM25 score (descending).

    Chunks with zero term overlap (score 0) are dropped so an off-topic
    query contributes nothing rather than arbitrary noise.
    """
    import numpy as np

    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(scores)[::-1][:k]
    return [int(i) for i in order if scores[i] > 0]


def dense_candidates(index, query_embedding, k: int) -> list[tuple[int, float]]:
    """Return up to `k` (chunk_index, squared-L2 distance) pairs, ascending."""
    import numpy as np

    query = np.asarray(query_embedding, dtype="float32")
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances, indices = index.search(query, k)
    return [(int(i), float(d)) for i, d in zip(indices[0], distances[0]) if i >= 0]


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Fuse several ranked id lists into one via reciprocal-rank fusion.

    Each item scores sum(1 / (k + rank)) across the lists it appears in
    (rank is 1-based). k=60 is the standard constant; no tuning needed.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def load_cross_encoder(model_name: str):
    """Lazily load a sentence-transformers CrossEncoder (heavy import)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _l2_distance(index, query_vec, idx: int) -> float:
    """Exact squared-L2 distance from query to a stored vector (for BM25-only
    hits that never appeared in the dense candidate pool)."""
    import numpy as np

    stored = index.reconstruct(int(idx))
    diff = np.asarray(query_vec, dtype="float32") - stored
    return float(diff @ diff)


def retrieve(
    index,
    chunks: list[dict],
    embedder,
    query: str,
    top_k: int = 4,
    *,
    hybrid: bool = False,
    rerank: bool = False,
    bm25=None,
    cross_encoder=None,
    candidate_k: int = 50,
    rrf_k: int = 60,
    max_distance: float | None = None,
) -> list[dict]:
    """Unified retrieval used by both the app and the eval harness.

    * dense-only (hybrid=False, rerank=False): identical to `search()`.
    * hybrid: fuse BM25 + dense candidates via RRF.
    * rerank: reorder the fused/dense pool with a cross-encoder.

    Thresholding is preserved sensibly: candidate generation ignores
    `max_distance` (so BM25 can rescue exact-term hits), but the final
    results are still gated by the dense squared-L2 distance, so an
    off-topic query — weak on both signals — yields an empty list.
    """
    query_emb = embed_query(embedder, query)

    if not hybrid and not rerank:
        return search(index, chunks, query_emb, top_k=top_k, max_distance=max_distance)

    dense = dense_candidates(index, query_emb, candidate_k)
    dist_by_idx = {idx: dist for idx, dist in dense}
    dense_order = [idx for idx, _ in dense]

    if hybrid:
        if bm25 is None:
            raise ValueError("hybrid=True requires a prebuilt bm25 (see build_bm25)")
        sparse_order = bm25_candidates(bm25, query, candidate_k)
        pool = reciprocal_rank_fusion([dense_order, sparse_order], k=rrf_k)
    else:
        pool = dense_order

    if rerank:
        if cross_encoder is None:
            raise ValueError("rerank=True requires a cross_encoder (see load_cross_encoder)")
        pool = pool[:candidate_k]
        if pool:
            scores = cross_encoder.predict([(query, chunks[i].get("chunk_text", "")) for i in pool])
            pool = [i for i, _ in sorted(zip(pool, scores), key=lambda kv: kv[1], reverse=True)]

    query_vec = query_emb.reshape(-1)
    results: list[dict] = []
    for idx in pool:
        dist = dist_by_idx.get(idx)
        if dist is None:
            dist = _l2_distance(index, query_vec, idx)
        if max_distance is not None and dist > max_distance:
            continue
        record = dict(chunks[idx])
        record["score"] = dist
        results.append(record)
        if len(results) >= top_k:
            break
    return results
