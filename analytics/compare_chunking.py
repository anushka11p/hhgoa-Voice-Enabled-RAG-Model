"""Score all four chunking strategies against the dataset's own labels.

This is what makes ACTIVE_CHUNK_STRATEGY an engineering decision rather than a
preference. For each strategy we build a complete index (chunk -> embed ->
FAISS + BM25 + centroids), run the same held-out queries through the same
hybrid retrieval, and score Recall@1 / Recall@5 / MRR plus the latency the
chunking choice implies.

The tradeoff being measured: smaller chunks localize the answer better
(higher precision per chunk) but produce a bigger index and dilute BM25
statistics; larger chunks retrieve more context per hit but bury the relevant
sentence. Recall@k against real query->passage labels is what settles it.

Run:  python -m analytics.compare_chunking --n 150
"""
import argparse
import json
import random
import statistics
import sys
import time

import config
from chunking.registry import STRATEGIES


def _reciprocal_rank(retrieved, relevant) -> float:
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(strategy: str, queries, rebuild: bool = True) -> dict:
    from data.build_index import build
    from retrieval.pipeline import RetrievalEngine

    index_name = f"cmp_{strategy}"
    t0 = time.perf_counter()
    if rebuild:
        manifest = build(strategy=strategy, index_name=index_name)
    else:
        manifest = {}
    build_s = time.perf_counter() - t0

    engine = RetrievalEngine(load_reranker=True, index_name=index_name)
    # Warm up so the first query is not charged to this strategy.
    engine.search(queries[0]["query"])

    r1, r5, mrr, lat = [], [], [], []
    for row in queries:
        relevant = set(row["relevant_passage_ids"])
        t = time.perf_counter()
        res = engine.search(row["query"])
        lat.append((time.perf_counter() - t) * 1000)
        got = [c.metadata.get("passage_id") for c in res.chunks]
        r1.append(1.0 if relevant & set(got[:1]) else 0.0)
        r5.append(1.0 if relevant & set(got[:5]) else 0.0)
        mrr.append(_reciprocal_rank(got, relevant))

    lat_sorted = sorted(lat)
    return {
        "strategy": strategy,
        "n_chunks": manifest.get("n_chunks"),
        "build_s": round(build_s, 1),
        "recall_at_1": round(statistics.mean(r1), 4),
        "recall_at_5": round(statistics.mean(r5), 4),
        "mrr": round(statistics.mean(mrr), 4),
        "retrieval_p50_ms": round(lat_sorted[len(lat_sorted) // 2], 2),
        "retrieval_p100_ms": round(lat_sorted[-1], 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    args = ap.parse_args()

    evals = json.loads(
        (config.DATA_DIR / "eval_queries.json").read_text(encoding="utf-8")
    )
    random.seed(7)
    queries = random.sample(evals, min(args.n, len(evals)))

    rows = []
    for strategy in args.strategies.split(","):
        print(f"\n=== {strategy} ===")
        rows.append(evaluate(strategy.strip(), queries))
        print(f"    {rows[-1]}")

    rows.sort(key=lambda r: -r["recall_at_5"])
    out = {
        "n_queries": len(queries),
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "results": rows,
        "winner": rows[0]["strategy"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = config.ANALYTICS_DIR / "chunking_comparison.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    hdr = f"| {'strategy':<10} | {'chunks':>6} | {'R@1':>6} | {'R@5':>6} | {'MRR':>6} | {'p50 ms':>7} |"
    print("\n" + hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for r in rows:
        print(f"| {r['strategy']:<10} | {r['n_chunks']:>6} | {r['recall_at_1']:>6.3f} | "
              f"{r['recall_at_5']:>6.3f} | {r['mrr']:>6.3f} | {r['retrieval_p50_ms']:>7.1f} |")
    print(f"\nWinner by Recall@5: {out['winner']}  ->  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
