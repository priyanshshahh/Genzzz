"""Build the FAISS index + metadata pair from a YouTube channel.

Replaces the original Colab notebook (youtube_scraper.ipynb) with a
config-driven, resumable CLI:

    python scripts/build_index.py                     # defaults from config.py
    python scripts/build_index.py --channel UCxxxx    # any public channel
    python scripts/build_index.py --limit 5           # smoke-test run

Captions are fetched with youtube-transcript-api; videos without captions
are skipped by default (pass --whisper to transcribe them locally with
openai-whisper, which is slow and needs ffmpeg).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core import pipeline, youtube


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default=config.CHANNEL_ID, help="YouTube channel id (UC...)")
    p.add_argument("--output-dir", type=Path, default=config.DATA_DIR, help="Where to write the index pair")
    p.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    p.add_argument("--embed-model", default=config.EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None, help="Only process the first N videos")
    p.add_argument("--whisper", action="store_true", help="Transcribe caption-less videos with Whisper (slow)")
    p.add_argument("--sleep-min", type=float, default=1.0, help="Min pause between transcript fetches")
    p.add_argument("--sleep-max", type=float, default=3.0, help="Max pause between transcript fetches")
    p.add_argument("--no-cache", action="store_true", help="Ignore the transcript cache and refetch everything")
    p.add_argument("--max-retries", type=int, default=3, help="Retries per video when YouTube rate-limits")
    p.add_argument("--backoff", type=float, default=60.0, help="Base seconds to wait after a rate-limit hit")
    return p.parse_args()


def fetch_with_backoff(video_id: str, max_retries: int, backoff: float) -> str | None:
    """Fetch a transcript, sleeping backoff * attempt on rate-limit hits."""
    for attempt in range(1, max_retries + 1):
        try:
            return youtube.fetch_transcript(video_id)
        except youtube.TranscriptFetchBlocked:
            if attempt == max_retries:
                raise
            wait = backoff * attempt
            print(f"        rate-limited; waiting {wait:.0f}s (attempt {attempt}/{max_retries}) ...")
            time.sleep(wait)
    return None


def load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    cache_path = out_dir / "transcripts_cache.json"

    print(f"[1/4] Listing uploads for channel {args.channel} ...")
    videos = youtube.fetch_channel_videos(args.channel)
    if not videos:
        print("No videos found — check the channel id / network.", file=sys.stderr)
        return 1
    if args.limit:
        videos = videos[: args.limit]
    print(f"      {len(videos)} videos")

    print("[2/4] Fetching transcripts (captions first, cache-aware) ...")
    cache: dict[str, str] = {} if args.no_cache else load_cache(cache_path)
    fetched = with_captions = blocked = 0
    for i, video in enumerate(videos, 1):
        vid = video["video_id"]
        if vid in cache:
            status = "cached" if cache[vid] else "cached (none)"
        else:
            try:
                text = fetch_with_backoff(vid, args.max_retries, args.backoff)
            except youtube.TranscriptFetchBlocked:
                # Do NOT cache — a rerun should retry these videos.
                blocked += 1
                print(f"      {i}/{len(videos)} {vid} BLOCKED (will retry on next run)")
                continue
            if text is None and args.whisper:
                print(f"      {i}/{len(videos)} {vid} no captions -> whisper ...")
                text = youtube.transcribe_with_whisper(vid)
            cache[vid] = text or ""
            save_cache(cache_path, cache)
            fetched += 1
            status = "ok" if text else "no transcript"
            time.sleep(random.uniform(args.sleep_min, args.sleep_max))
        if cache.get(vid):
            with_captions += 1
        print(f"      {i}/{len(videos)} {vid} {status}")
    print(f"      transcripts: {with_captions}/{len(videos)} videos "
          f"({fetched} newly fetched, {blocked} blocked)")
    if blocked:
        print(f"      WARNING: {blocked} videos were rate-limited by YouTube — "
              "rerun later to fetch them; the cache keeps what succeeded.")

    print("[3/4] Cleaning + chunking ...")
    chunks = pipeline.chunk_corpus(videos, cache, args.chunk_size, args.chunk_overlap)
    if not chunks:
        print("No chunks produced — no usable transcripts.", file=sys.stderr)
        return 1
    print(f"      {len(chunks)} chunks from {with_captions} transcribed videos")

    print(f"[4/4] Embedding with {args.embed_model} + writing FAISS pair ...")
    embedder = pipeline.get_embedder(args.embed_model)
    embeddings = pipeline.embed_texts(embedder, [c["chunk_text"] for c in chunks], args.batch_size)
    index = pipeline.build_faiss_index(embeddings)
    build_info = {
        "channel_id": args.channel,
        "embed_model": args.embed_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_videos": len(videos),
        "num_videos_with_transcript": with_captions,
        "whisper_fallback": bool(args.whisper),
    }
    index_path = out_dir / "faiss_index.index"
    metadata_path = out_dir / "faiss_metadata.json"
    pipeline.write_index_pair(index, chunks, index_path, metadata_path, build_info)
    print(f"      wrote {index_path} ({index.ntotal} vectors, dim={index.d})")
    print(f"      wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
