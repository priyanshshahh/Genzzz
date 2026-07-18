"""Central configuration for ChannelMind.

Every value here is a default; the build and eval scripts accept CLI flags
that override them. Paths are resolved relative to this file so the app
works from any working directory (locally, CI, or HF Spaces).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

# Corpus source: YouTube channel ID (the "UC..." form).
# Default channel: Aprilynne Alter (YouTube-growth educator, captioned uploads).
CHANNEL_ID = "UC-PaZZpjgJ61wkK9yKfpe8w"

# Chunking (RecursiveCharacterTextSplitter)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embeddings — all-MiniLM-L6-v2 outputs unit-normalised 384-dim vectors,
# so FAISS squared-L2 distance = 2 - 2*cosine_similarity, bounded in [0, 4].
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 64

# Generation. gemini-2.5-flash is rejected for new API keys (404), so the
# default is the current stable flash model.
GEMINI_MODEL = "gemini-3.5-flash"
# A grounded RAG answer should stick to the retrieved excerpts, so keep the
# sampling temperature low (deterministic-ish) and cap the output length.
GEN_TEMPERATURE = 0.2
GEN_MAX_OUTPUT_TOKENS = 1024

# Index artifacts (written/read as a matched pair)
INDEX_PATH = DATA_DIR / "faiss_index.index"
METADATA_PATH = DATA_DIR / "faiss_metadata.json"
TRANSCRIPT_CACHE_PATH = DATA_DIR / "transcripts_cache.json"

# Retrieval. Squared-L2 threshold above which a chunk is considered
# irrelevant (1.10 ~= cosine similarity 0.45). Tuned on the labeled set in
# eval/queries.json — see scripts/eval_retrieval.py --report-distances.
DISTANCE_THRESHOLD = 1.10
TOP_K = 4

# Per-chunk excerpt length fed into the Gemini prompt. Chunks are built at
# CHUNK_SIZE characters, so show the whole chunk rather than silently
# dropping its tail — this is deliberately aligned to CHUNK_SIZE, not a
# magic number disconnected from the build config.
EXCERPT_CHARS = CHUNK_SIZE

# --- Hybrid retrieval + reranking -----------------------------------------
# Dense-only is the honest baseline; these flags let the app and the eval
# harness A/B the production-standard additions on the same committed index.
#
# HYBRID: fuse BM25 (sparse/keyword) with dense FAISS via reciprocal-rank
# fusion, so exact terms (proper nouns, product names, numbers) the dense
# model ranks too low still surface.
# RERANK: reorder the fused candidate pool with a CPU cross-encoder.
HYBRID = True
RERANK = True

# Reciprocal-rank-fusion constant (standard default; no tuning needed).
RRF_K = 60
# How many chunks each retriever contributes to the fused candidate pool,
# and how many the cross-encoder scores before the final top_k cut.
CANDIDATE_K = 50
# Cross-encoder reranker (small, CPU-feasible, ~90 MB download on first use;
# lazy-loaded so a dense-only or hybrid-only deploy never pays for it).
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
