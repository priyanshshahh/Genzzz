"""Evaluate retrieval quality on a labeled question -> expected-video set.

Metrics (video-level): Recall@1, Recall@3, Recall@5 and MRR. A query's
retrieved video ranking is obtained by searching chunk vectors and keeping
each video's best (lowest) distance.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --report-distances   # threshold tuning aid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core.pipeline import get_embedder
from core.retrieval import load_index_pair, search

EVAL_SET_PATH = config.REPO_ROOT / "eval" / "queries.json"

# Deliberately off-topic queries used only for the distance report — they
# should land ABOVE the threshold so the app answers "no relevant content".
OFFTOPIC_QUERIES = [
    "What is the boiling point of liquid nitrogen?",
    "Best pasta recipe for a quick dinner",
    "How do I file taxes as a freelancer in Germany?",
    "Explain the rules of cricket",
    "What year did the Roman Empire fall?",
]


def video_ranking(index, chunks, query_emb, depth: int = 50) -> list[str]:
    """Rank video_ids by their best-chunk distance for a query."""
    results = search(index, chunks, query_emb, top_k=depth, max_distance=None)
    ranked = []
    for r in results:  # results are distance-ascending
        vid = r["video_id"]
        if vid not in ranked:
            ranked.append(vid)
    return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH)
    parser.add_argument("--index", type=Path, default=config.INDEX_PATH)
    parser.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    parser.add_argument("--report-distances", action="store_true")
    args = parser.parse_args()

    with open(args.eval_set, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    index, metadata = load_index_pair(args.index, args.metadata)
    chunks = metadata["chunks"]
    embedder = get_embedder(config.EMBED_MODEL)

    ranks: list[int | None] = []
    top1_distances = []
    print(f"{len(eval_set)} labeled queries against {index.ntotal} chunks\n")
    for item in eval_set:
        emb = embedder.encode([item["question"]]).astype("float32")
        ranked = video_ranking(index, chunks, emb)
        expected = item.get("expected_video_ids") or [item["expected_video_id"]]
        hits = [ranked.index(v) + 1 for v in expected if v in ranked]
        rank = min(hits) if hits else None
        ranks.append(rank)
        best = search(index, chunks, emb, top_k=1, max_distance=None)
        top1_distances.append(best[0]["score"] if best else float("nan"))
        marker = f"rank {rank}" if rank else "MISS"
        print(f"  [{marker:>7}] d1={top1_distances[-1]:.3f}  {item['question']}")

    n = len(ranks)
    recall_at = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5)}
    mrr = sum(1 / r for r in ranks if r is not None) / n

    print("\n=== Retrieval metrics (video-level) ===")
    for k in (1, 3, 5):
        print(f"Recall@{k}: {recall_at[k]:.3f}")
    print(f"MRR:      {mrr:.3f}")

    if args.report_distances:
        print("\n=== Distance report (squared L2; threshold tuning) ===")
        print(f"On-topic top-1 distances:  min={min(top1_distances):.3f} "
              f"max={max(top1_distances):.3f} mean={sum(top1_distances)/n:.3f}")
        print("Off-topic queries:")
        off = []
        for q in OFFTOPIC_QUERIES:
            emb = embedder.encode([q]).astype("float32")
            best = search(index, chunks, emb, top_k=1, max_distance=None)
            off.append(best[0]["score"])
            print(f"  d1={best[0]['score']:.3f}  {q}")
        print(f"Off-topic top-1 distances: min={min(off):.3f} max={max(off):.3f}")
        print(f"Configured threshold: {config.DISTANCE_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
