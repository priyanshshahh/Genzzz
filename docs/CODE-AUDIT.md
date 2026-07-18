# Code Audit — ChannelMind

Honest, as-is audit of the codebase at commit `5bee585` (branch `production`).
Document-only: no fixes applied here. Findings are ranked roughly by
value-to-fix (impact vs. effort), not by severity alone. Market/competitive
comparison (e.g. "is MiniLM the right choice vs. X") is out of scope for
this document — only what's directly observable in the code is noted here.

## Findings, ranked

### 1. Retrieval quality ceiling — dense-only, no hybrid/rerank (high value, high effort)
The repo's own eval harness (`scripts/eval_retrieval.py`) reports Recall@1
0.250, Recall@3/@5 0.450, MRR 0.365 on the 20-query labeled set against the
committed 1141-chunk index. 9 of 20 queries never surface the expected
video in the top 50 nearest chunks at all. This is a single `IndexFlatL2`
search over MiniLM embeddings with no BM25/keyword hybrid and no
cross-encoder reranking stage — `core/retrieval.py::search` does one FAISS
call and returns. The README already discloses this as "the honest
baseline"; it's the largest lever on answer quality in the whole system and
the only one with a real, reproducible number attached.

### 2. No error handling around the Gemini call in `app.py` (high value, trivial effort)
Every other failure path in `app.py` is deliberately guarded: missing index
artifacts (`FileNotFoundError`), inconsistent pair (`IndexMetadataMismatch`),
and no-results-under-threshold all produce a friendly `st.error`/`st.warning`.
`llm.invoke(build_rag_prompt(results, user_question))` (line ~110) has no
`try/except` at all. Any Gemini-side failure — quota exhaustion, transient
network error, an API-side content/safety block, a malformed key at request
time — will raise inside `st.spinner("Generating answer...")` and surface
as a raw Streamlit exception/traceback to the end user, which is a jarring
inconsistency given how carefully every other error path is handled.

### 3. Prompt truncation magic number disagrees with the chunk-size config (medium-high value, trivial effort)
`build_rag_prompt` in `app.py` truncates every retrieved chunk to
`c.get('chunk_text', '')[:800]` before it goes into the Gemini prompt, but
`config.CHUNK_SIZE = 1000` is the size chunks were actually built at. Up to
200 characters (20%) of every retrieved chunk's tail is silently dropped
from what the LLM sees, and `800` appears nowhere in `config.py` — it's a
literal in `app.py` disconnected from the single source of truth the rest
of the codebase otherwise respects. If chunk size is ever tuned in
`config.py`, this number won't follow it.

### 4. Retrieval eval harness is not wired into CI (medium value, low-medium effort)
`.github/workflows/ci.yml` runs only `pytest tests/` on every push/PR to
`main`/`production`. `scripts/eval_retrieval.py` — the only thing that
actually measures retrieval quality (Recall@k, MRR, threshold separation)
— is never invoked in CI. A change to chunking, the embedding model, or the
distance threshold could silently regress real retrieval quality with
nothing failing red; the 29 pytest tests only check pipeline/plumbing
correctness (chunking mechanics, index pairing, threshold filtering logic
on synthetic fixtures), not retrieval quality against the real labeled set.

### 5. Fixed-size character chunking on conversational transcript text (medium-high value, medium-high effort)
`chunk_transcript` (`core/pipeline.py`) uses
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,
separators=["\n\n","\n",".","!","?",",", " ", ""])` uniformly across every
video. The committed corpus is conversational/interview transcripts with
filler and cross-talk (see sample chunk text in `data/faiss_metadata.json`,
e.g. "Um... Amazing. Thank you so much for having me") rather than
structured prose — there's no topic-boundary detection, no filler-word
stripping beyond `clean_text`'s timestamp/URL/whitespace normalization, and
no per-video adjustment (a 40-minute interview and a 3-minute tutorial get
chunked identically). This is a plausible contributor to the Recall@1
weakness in finding #1, though disentangling "chunking" from "embedding
model" from "no rerank" as the dominant cause would need controlled
experiments this audit doesn't run.

### 6. `schema_version` is written but never validated on read (medium value, low effort)
`core/pipeline.py` defines `METADATA_SCHEMA_VERSION = 1` and writes it into
every `faiss_metadata.json` under `schema_version`. `core/retrieval.py`'s
`load_index_pair` — the only reader — checks `ntotal == len(chunks)` and
`index_dim`, but never reads or compares `schema_version`. The versioning
mechanism exists on the write side with no enforcement on the read side, so
a future incompatible metadata shape (e.g. renamed/removed chunk fields)
carrying an old `schema_version` would load without complaint and likely
fail later with a confusing `KeyError`/`AttributeError` deep in `app.py`
instead of the clear "rebuild with scripts/build_index.py" message the
pairing checks already produce for the other two invariants.

### 7. Dead code: legacy Colab notebook duplicates the production pipeline (medium value, low effort)
`notebooks/youtube_scraper.ipynb` (1039 lines, 17 cells) is the original
notebook that `scripts/build_index.py` + `core/*` replaced (per
`docs/PROJECT-NOTES.md`'s 2026-07-12 changelog entry). It still contains
its own inferior copies of transcript fetching, `clean_text`-equivalent
regex cleaning, chunking via `langchain.text_splitter` (not
`langchain_text_splitters`), embedding, and FAISS index construction, all
against hardcoded `/content/drive/MyDrive/...` Colab paths with no error
handling. It's committed with no marker (README note, `docs/` pointer, or
`archive/` relocation) distinguishing it as historical, so a new contributor
skimming the repo could reasonably mistake it for a second, parallel build
path or copy stale patterns from it.

### 8. `transcribe_with_whisper`'s blanket `except Exception` masks real failures (medium value, low effort)
In `core/youtube.py`, the Whisper fallback path wraps the entire
download+transcribe body in `try: ... except Exception: return None`. A
genuinely silent video (no speech) and a corrupted audio download, a
missing/broken ffmpeg install, a Whisper model load failure, or an
out-of-memory crash are all indistinguishable — every case returns `None`
and gets cached in `transcripts_cache.json` as `""`, identical to "this
video really has no transcribable audio." Since `--whisper` is the fallback
specifically meant to rescue caption-less videos, silently losing the
distinction between "nothing to transcribe" and "transcription broke"
undermines the one thing this path exists for.

### 9. No generation-time configuration surface (low-medium value, low effort)
`ChatGoogleGenerativeAI(model=config.GEMINI_MODEL)` in `app.py` is
constructed with no `temperature`, `max_output_tokens`, or
`safety_settings` — every other tunable in the system (chunk size/overlap,
embed model, batch size, distance threshold, top_k) lives in `config.py`
with a documented rationale, but generation behavior is entirely whatever
the SDK's defaults happen to be, with no equivalent config knobs. This is
an asymmetry in an otherwise consistently config-driven codebase, not a
bug — the app works fine with defaults today.

### 10. Rate-limit backoff has no cross-run memory (low-medium value, medium effort)
`scripts/build_index.py`'s retry/backoff (`--max-retries`, `--backoff`,
`--sleep-min/--sleep-max`) is entirely local to a single process
invocation. Blocked videos are correctly left out of the cache so a rerun
retries them (a real strength — see `docs/PROJECT-NOTES.md`), but nothing
persists *when* the last block happened, so a user who reruns immediately
after a blocked run has no automated protection against being blocked again
within the same minute; avoiding that relies entirely on the operator
manually waiting and tuning `--sleep-min/--sleep-max` between runs.

## Not counted as findings (verified, not issues)

- `response.text` in `app.py` (the `.content` vs `.text` comment) was
  checked against the pinned `langchain-core==1.4.9` (transitively via
  `langchain-google-genai==4.2.7` in `requirements.txt`) actually installed
  in `.venv/` — `.text` is a valid property on `AIMessage` at that version,
  confirmed by direct import. Not a bug.
- Single-channel index / no incremental rebuild, captions-only by default,
  and "pure dense, no hybrid/rerank" are already disclosed in the README's
  own Limitations section — restated above only where the eval numbers add
  something the README doesn't already say plainly.
