"""Loading the index/metadata pair and thresholded similarity search."""

from __future__ import annotations

import json
from pathlib import Path


class IndexMetadataMismatch(RuntimeError):
    """The FAISS index and metadata JSON do not describe the same build."""


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
