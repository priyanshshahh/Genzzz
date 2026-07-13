# Project Notes — ChannelMind (repo: Genzzz)

## Changelog

### 2026-07-12 — Production rework
- Backed up original state on branch `pre-campaign` (single Colab commit).
- Converted `youtube_scraper.ipynb` (Colab) into a scripted pipeline:
  `scripts/build_index.py`, config-driven via `config.py` + argparse.
  Whisper audio transcription is now opt-in (`--whisper`); default build
  uses captions only (youtube-transcript-api) for speed.
- Fixed `app.py`: hardcoded `~/Downloads/...` paths replaced with
  repo-relative `pathlib` paths; missing/mismatched index artifacts now
  produce an `st.error` with build instructions instead of a traceback.
- Added FAISS squared-L2 relevance threshold (`config.DISTANCE_THRESHOLD`)
  so off-topic questions return "no relevant content" instead of forced
  top-k neighbours.
- Committed the previously-missing `faiss_metadata.json` as a matched pair
  with the index, regenerated from a real channel scrape (the original
  repo shipped the index binary but never the metadata, so retrieval could
  not work at all).
- Standardized generation on `gemini-3.5-flash` (the old default
  `gemini-1.5-pro-latest` is retired, and `gemini-2.5-flash` now returns
  404 for new API keys).
- Pinned all dependencies (`requirements.txt` runtime / `requirements-dev.txt`
  build+test) from a working Python 3.12 venv.
- Added offline pytest suite (chunking, index build on fixture corpus,
  pairing invariant, threshold behavior, mocked YouTube calls) + GitHub
  Actions CI.
- Added `scripts/eval_retrieval.py` with a 20-question labeled set derived
  from real video titles; metrics in README are from a real run.

## Operational notes
- The transcript endpoint aggressively IP-blocks bulk fetching (~20-25
  rapid requests). `build_index.py` treats `RequestBlocked` as retryable,
  never caches it as "no transcript", and resumes from
  `data/transcripts_cache.json` on rerun. Use `--sleep-min/--sleep-max`
  generously for full-channel builds.
- Index artifacts (`data/faiss_index.index` + `data/faiss_metadata.json`)
  are a matched pair; loaders enforce `ntotal == len(chunks)` and the
  declared embedding dimension.
- Secrets: only `GOOGLE_API_KEY`, via Streamlit secrets or env var. Never
  committed.
