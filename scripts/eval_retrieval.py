"""Evaluate retrieval quality on a labeled question -> expected-video set.

Metrics (video-level): Recall@1, Recall@3, Recall@5 and MRR. A query's
retrieved video ranking is obtained by searching chunk vectors and keeping
each video's best (lowest) rank.

    python scripts/eval_retrieval.py                       # dense-only baseline
    python scripts/eval_retrieval.py --hybrid              # + BM25/RRF fusion
    python scripts/eval_retrieval.py --hybrid --rerank     # + cross-encoder
    python scripts/eval_retrieval.py --report-distances    # threshold tuning aid

CI regression gate (fail if the committed index regresses below a floor):

    python scripts/eval_retrieval.py --min-r1 0.25 --min-r5 0.45 --min-mrr 0.36
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core.pipeline import get_embedder
from core.retrieval import (
    build_bm25,
    dense_candidates,
    load_cross_encoder,
    load_index_pair,
    retrieve,
)

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


def video_ranking(records: list[dict]) -> list[str]:
    """Collapse ranked chunk records to a video-level ranking (best rank first)."""
    ranked: list[str] = []
    for r in records:
        vid = r["video_id"]
        if vid not in ranked:
            ranked.append(vid)
    return ranked


def best_dense_distance(index, embedder, question: str) -> float:
    """Top-1 dense squared-L2 distance (the signal the threshold gates on),
    computed independently of the retrieval mode."""
    from core.retrieval import embed_query

    hits = dense_candidates(index, embed_query(embedder, question), 1)
    return hits[0][1] if hits else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=EVAL_SET_PATH)
    parser.add_argument("--index", type=Path, default=config.INDEX_PATH)
    parser.add_argument("--metadata", type=Path, default=config.METADATA_PATH)
    parser.add_argument("--hybrid", action="store_true", help="Fuse BM25 with dense (RRF)")
    parser.add_argument("--rerank", action="store_true", help="Rerank the fused pool with a cross-encoder")
    parser.add_argument("--depth", type=int, default=50, help="Ranking depth for Recall@k")
    parser.add_argument("--report-distances", action="store_true")
    parser.add_argument("--min-r1", type=float, default=None, help="Fail if Recall@1 below this")
    parser.add_argument("--min-r3", type=float, default=None, help="Fail if Recall@3 below this")
    parser.add_argument("--min-r5", type=float, default=None, help="Fail if Recall@5 below this")
    parser.add_argument("--min-mrr", type=float, default=None, help="Fail if MRR below this")
    args = parser.parse_args()

    with open(args.eval_set, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    index, metadata = load_index_pair(args.index, args.metadata)
    chunks = metadata["chunks"]
    embedder = get_embedder(config.EMBED_MODEL)
    bm25 = build_bm25(chunks) if args.hybrid else None
    cross_encoder = load_cross_encoder(config.CROSS_ENCODER_MODEL) if args.rerank else None

    mode = "dense"
    if args.hybrid:
        mode = "hybrid+rerank" if args.rerank else "hybrid"
    elif args.rerank:
        mode = "dense+rerank"

    ranks: list[int | None] = []
    top1_distances = []
    print(f"{len(eval_set)} labeled queries against {index.ntotal} chunks  [mode: {mode}]\n")
    for item in eval_set:
        records = retrieve(
            index, chunks, embedder, item["question"], top_k=args.depth,
            hybrid=args.hybrid, rerank=args.rerank, bm25=bm25,
            cross_encoder=cross_encoder, candidate_k=args.depth, max_distance=None,
        )
        ranked = video_ranking(records)
        expected = item.get("expected_video_ids") or [item["expected_video_id"]]
        hits = [ranked.index(v) + 1 for v in expected if v in ranked]
        rank = min(hits) if hits else None
        ranks.append(rank)
        top1_distances.append(best_dense_distance(index, embedder, item["question"]))
        marker = f"rank {rank}" if rank else "MISS"
        print(f"  [{marker:>7}] d1={top1_distances[-1]:.3f}  {item['question']}")

    n = len(ranks)
    recall_at = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5)}
    mrr = sum(1 / r for r in ranks if r is not None) / n

    print(f"\n=== Retrieval metrics (video-level, mode: {mode}) ===")
    for k in (1, 3, 5):
        print(f"Recall@{k}: {recall_at[k]:.3f}")
    print(f"MRR:      {mrr:.3f}")
    print(f"Targets never in top-{args.depth}: {sum(1 for r in ranks if r is None)}/{n}")

    if args.report_distances:
        print("\n=== Distance report (squared L2; threshold tuning) ===")
        print(f"On-topic top-1 distances:  min={min(top1_distances):.3f} "
              f"max={max(top1_distances):.3f} mean={sum(top1_distances)/n:.3f}")
        print("Off-topic queries:")
        off = []
        for q in OFFTOPIC_QUERIES:
            d = best_dense_distance(index, embedder, q)
            off.append(d)
            print(f"  d1={d:.3f}  {q}")
        print(f"Off-topic top-1 distances: min={min(off):.3f} max={max(off):.3f}")
        print(f"Configured threshold: {config.DISTANCE_THRESHOLD}")

    # Regression gate: fail (nonzero exit) if any provided floor is breached.
    floors = {"Recall@1": (args.min_r1, recall_at[1]), "Recall@3": (args.min_r3, recall_at[3]),
              "Recall@5": (args.min_r5, recall_at[5]), "MRR": (args.min_mrr, mrr)}
    failures = [f"{name} {val:.3f} < floor {floor:.3f}"
                for name, (floor, val) in floors.items() if floor is not None and val < floor]
    if failures:
        print("\nREGRESSION GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if any(floor is not None for floor, _ in floors.values()):
        print("\nRegression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
