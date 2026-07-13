"""Transcript cleaning, chunking, embedding, and FAISS index construction."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from core.youtube import VIDEO_FIELDS

_TIMESTAMP_RE = re.compile(r"\[?\d{1,2}:\d{2}(?::\d{2})?\]?")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")

METADATA_SCHEMA_VERSION = 1


def clean_text(text) -> str:
    """Strip timestamps, URLs, and collapse whitespace. Non-strings -> ''."""
    if not isinstance(text, str):
        return ""
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = _WS_RE.sub(" ", text)
    return text.strip()


def chunk_transcript(transcript: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split a cleaned transcript into overlapping chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    return splitter.split_text(transcript)


def chunk_corpus(
    videos: list[dict],
    transcripts: dict[str, str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Build chunk records from videos + their transcripts.

    Videos without a usable transcript are skipped. Each record carries the
    lean video metadata plus `chunk_index` / `chunk_text`.
    """
    records = []
    for video in videos:
        raw = transcripts.get(video["video_id"]) or ""
        cleaned = clean_text(raw)
        if not cleaned:
            continue
        for idx, chunk in enumerate(chunk_transcript(cleaned, chunk_size, chunk_overlap)):
            record = {field: video.get(field) for field in VIDEO_FIELDS}
            record["chunk_index"] = idx
            record["chunk_text"] = chunk
            records.append(record)
    return records


def get_embedder(model_name: str):
    """Load the sentence-transformers model (heavy import kept lazy)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(embedder, texts: list[str], batch_size: int = 64):
    """Embed texts into a float32 numpy array of shape (n, dim)."""
    import numpy as np

    embeddings = embedder.encode(
        texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings):
    """Build an exact (IndexFlatL2) FAISS index from an (n, dim) array."""
    import faiss

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Expected a non-empty (n, dim) array, got shape {embeddings.shape}")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def write_index_pair(
    index,
    chunks: list[dict],
    index_path: Path,
    metadata_path: Path,
    build_info: dict | None = None,
) -> None:
    """Atomically-ish write the FAISS index and its metadata as a matched pair.

    The metadata JSON records the chunk list plus build provenance; loaders
    verify that `index.ntotal == len(chunks)` before serving queries.
    """
    import faiss

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Refusing to write mismatched pair: index has {index.ntotal} vectors "
            f"but {len(chunks)} metadata chunks"
        )
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "build_info": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_chunks": len(chunks),
            "index_dim": index.d,
            **(build_info or {}),
        },
        "chunks": chunks,
    }
    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
