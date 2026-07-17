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

### 2026-07-17 — Real retrieval eval + live Gemini verify
- Ran `scripts/eval_retrieval.py --report-distances` against the committed
  `data/faiss_index.index`/`faiss_metadata.json` pair (1141 chunks) on the
  20-question labeled set in `eval/queries.json`, in a Python 3.12 `uv`
  venv (`.venv/`, already in the repo). Real results, now in the README:
  Recall@1 0.250, Recall@3 0.450, Recall@5 0.450, MRR 0.365. 9/20 queries
  never surface the expected video in the top 50 — the honest baseline for
  plain dense MiniLM retrieval with no hybrid/rerank. The distance
  threshold (1.10) does cleanly separate on-topic (0.467-1.073) from
  off-topic (1.458-1.688) queries on this run, so the "no relevant
  content" fallback is working as designed even though in-topic recall
  has real headroom.
- Ran the full pytest suite: 29/29 passed.
- Live end-to-end verify of the actual RAG + Gemini path (retrieval ->
  prompt build -> `ChatGoogleGenerativeAI(model="gemini-3.5-flash").invoke`,
  same code app.py runs), with `GOOGLE_API_KEY` sourced at runtime from
  `hiring/.secrets/campaign-keys.env` (never written to any tracked file).
  Question: "How much money can you make in your first year on YouTube?"
  Retrieval returned 4 chunks under the 1.10 threshold, top hit distance
  0.584, all from the two videos actually about first-year earnings.
  Gemini's answer: cited both source videos with titles/URLs, explained
  the YPP requirements (1,000 subs / 4,000 watch hours) and RPM concept
  from the transcript, cited the $290,000/7-income-streams figure from the
  second video, and explicitly stated the excerpts don't contain "the
  exact final dollar amount" for the first year rather than inventing one.
  Confirms grounded generation and honest non-hallucination behavior work
  as designed.
- HF Spaces deploy prep: README front-matter (`title`/`emoji`/`sdk:
  streamlit`/`sdk_version`/`app_file: app.py`) is already the Spaces config
  block, `requirements.txt` is the runtime-only pinned set Spaces will
  install, and `data/faiss_index.index` + `data/faiss_metadata.json` are
  committed so the Space needs no YouTube access at runtime. Nothing else
  is required on the repo side. **Deploy itself is not executed** — it
  needs the owner's one-time HF login/Space creation and setting
  `GOOGLE_API_KEY` under Settings -> Variables and secrets on the Space.

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
