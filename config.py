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

# Generation
GEMINI_MODEL = "gemini-2.5-flash"

# Index artifacts (written/read as a matched pair)
INDEX_PATH = DATA_DIR / "faiss_index.index"
METADATA_PATH = DATA_DIR / "faiss_metadata.json"
TRANSCRIPT_CACHE_PATH = DATA_DIR / "transcripts_cache.json"

# Retrieval. Squared-L2 threshold above which a chunk is considered
# irrelevant (1.10 ~= cosine similarity 0.45). Tuned on the labeled set in
# eval/queries.json — see scripts/eval_retrieval.py --report-distances.
DISTANCE_THRESHOLD = 1.10
TOP_K = 4
