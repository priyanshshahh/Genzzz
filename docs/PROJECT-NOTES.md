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

### 2026-07-17 — Retrieval engineering: hybrid + rerank, expanded eval, CI gate

Two-phase pass over the CODE-AUDIT findings and the campaign-3 benchmark.
All numbers below are from real runs this session in the repo `.venv/`
(Python 3.12), `PYTHONHASHSEED=0`.

**Phase A — audit fixes**
- Wrapped the Gemini `llm.invoke()` in `app.py` in try/except → friendly
  `st.error` instead of a raw traceback (matches every other guarded path).
- Replaced the magic `[:800]` prompt-excerpt truncation with
  `config.EXCERPT_CHARS` (= `CHUNK_SIZE` = 1000) so the whole chunk reaches
  the LLM and the number tracks the build config.
- `core/retrieval.load_index_pair` now validates `schema_version` against
  `SUPPORTED_SCHEMA_VERSION` and fails with a clear rebuild message.
- `transcribe_with_whisper` no longer swallows every error as "no
  transcript": genuine silence → `None`; a real failure (bad download,
  missing ffmpeg, model-load/OOM) is logged and raised as
  `WhisperTranscriptionError`, which `build_index.py` treats as retryable
  (not cached), like a rate-limit block.
- Added a generation config surface: `config.GEN_TEMPERATURE = 0.2`,
  `config.GEN_MAX_OUTPUT_TOKENS = 1024`, passed into `ChatGoogleGenerativeAI`.
- Deleted the dead 1039-line Colab notebook (`notebooks/youtube_scraper.ipynb`);
  `scripts/` + `core/` fully supersede it.
- Wired `scripts/eval_retrieval.py` into CI as a regression gate (new
  `eval-gate` job): installs the real MiniLM embedder, runs dense-only on the
  legacy set, and fails if it drops below the published floor
  (`--min-r1 0.25 --min-r5 0.45 --min-mrr 0.36`). The gate flags are new
  args on the eval script; exit code 1 on breach.

**Phase B — build (hybrid → rerank → expanded eval → before/after)**
- Hybrid retrieval: `rank_bm25` (BM25Okapi) over the same chunks, fused with
  dense FAISS via reciprocal-rank fusion (`RRF_K = 60`). `config.HYBRID`.
- Cross-encoder reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` over the
  fused top-50 (`CANDIDATE_K = 50`), CPU, lazy-loaded only when
  `config.RERANK` is on. Both flags default **on** (best config below).
- Unified `core.retrieval.retrieve()` used by both `app.py` and the eval
  harness so they never drift; thresholding preserved (candidate generation
  ignores the distance gate so BM25 can rescue exact-term hits, but final
  results are still gated by dense squared-L2 → off-topic → empty).
- Expanded eval set: `eval/queries_expanded.json`, 55 queries authored
  against the actual transcripts, all 55 targets in-corpus (0 unreachable).
  Legacy `eval/queries.json` (20) kept unchanged as the published baseline.

**Exact commands + results**

```
# baseline / hybrid / hybrid+rerank on the legacy-20 set
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --hybrid
python scripts/eval_retrieval.py --hybrid --rerank
# same three on the expanded-55 set
python scripts/eval_retrieval.py --eval-set eval/queries_expanded.json [--hybrid [--rerank]]
```

| Set | Config | R@1 | R@3 | R@5 | MRR | unreachable |
|-----|--------|-----|-----|-----|-----|-------------|
| legacy-20 | dense           | 0.250 | 0.450 | 0.450 | 0.365 | 9/20 |
| legacy-20 | hybrid          | 0.350 | 0.450 | 0.550 | 0.425 | 9/20 |
| legacy-20 | hybrid+rerank   | 0.350 | 0.450 | 0.550 | 0.412 | 9/20 |
| exp-55    | dense           | 0.509 | 0.818 | 0.855 | 0.684 | 0/55 |
| exp-55    | hybrid          | 0.509 | 0.855 | 0.909 | 0.688 | 0/55 |
| exp-55    | hybrid+rerank   | 0.673 | 0.891 | 0.964 | 0.788 | 0/55 |

Honesty calls:
- **Legacy hybrid+rerank is an honest negative**: rerank leaves Recall@k flat
  and *lowers* MRR (0.425 → 0.412) vs. hybrid alone. Reported as-is. Root
  cause is the legacy set's 9/20 unreachable targets making n effectively 11
  and the reranker's reshuffle pure noise — the reason the set was expanded.
- On the trustworthy expanded set the win is real and the reranker is the
  dominant lever (R@1 +16.4pp, MRR +10.4pp over hybrid). Shipped as default.
- Expanded numbers are **never** presented as comparable to the old
  published legacy numbers; both tables are labeled and the legacy set's
  unreachable-target cap is stated.

**Verify**
- `pytest tests/ -q` → 44 passed (29 original + 1 schema_version test + 14
  new hybrid/RRF/rerank tests using the fake embedder + a fake cross-encoder).
- Streamlit app boots headless (HTTP 200, `/_stcore/health` ok); the full
  retrieve→prompt path exercised directly with the shipped hybrid+rerank
  config against the real index.
- `requirements.txt` adds pinned `rank-bm25==0.2.2`; the cross-encoder reuses
  the already-pinned `sentence-transformers`. HF Spaces cold-start note: the
  ~90 MB cross-encoder downloads on first query when `RERANK=True`.
- Query expansion (benchmark item 5/HyDE) was **not** built this session —
  left as a future measured experiment; the harness now supports testing it
  honestly.

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
