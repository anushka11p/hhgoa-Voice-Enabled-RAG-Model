"""End-to-end latency + retrieval-quality benchmark.

Runs N real queries sampled from the dataset's own eval split through the full
RAG core and reports P50 / P70 / P100 for every stage, plus Recall@k and MRR
scored against the dataset's query -> passage labels.

Two things this is careful about:

  * It warms up first. A cold first query costs seconds of lazy model init.
    Including that in a percentile would make P100 a measure of process
    startup, not of the pipeline.
  * It separates the RAG core from speech-to-text. STT is an HTTPS round trip
    to Sarvam; it is measured by --with-stt against the real sample audio and
    reported on its own, never folded into the core percentiles.

Run:  python -m analytics.run_benchmark --n 100
      python -m analytics.run_benchmark --n 100 --with-stt
"""
import argparse
import json
import random
import statistics
import sys
import time
from typing import Dict, List

import config
from harness.orchestrator import run_from_text
from harness.schemas import PipelineResult


def pct(values: List[float], p: float) -> float:
    """Nearest-rank percentile. p100 is the true worst observed case."""
    if not values:
        return 0.0
    s = sorted(values)
    if p >= 100:
        return round(s[-1], 3)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return round(s[k], 3)


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "p50": pct(values, 50),
        "p70": pct(values, 70),
        "p90": pct(values, 90),
        "p95": pct(values, 95),
        "p100": pct(values, 100),
        "mean": round(statistics.mean(values), 3),
        "n": len(values),
    }


def _reciprocal_rank(retrieved: List[str], relevant: set) -> float:
    for rank, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def run(n: int = 100, mode: str = None, seed: int = 42) -> dict:
    from retrieval.pipeline import warmup

    eval_path = config.DATA_DIR / "eval_queries.json"
    if not eval_path.exists():
        raise SystemExit("Run `python -m data.build_corpus` first.")
    evals = json.loads(eval_path.read_text(encoding="utf-8"))

    random.seed(seed)
    sample = random.sample(evals, min(n, len(evals)))

    print(f"Warming up models ...")
    warmup()

    print(f"Running {len(sample)} queries (mode={mode or config.GENERATION_MODE}) ...")
    core, total, stages = [], [], {}
    recall_5, recall_1, mrr = [], [], []
    refused = {"total": 0}
    over_budget = 0
    rerank_skipped = 0
    records = []

    for i, row in enumerate(sample, 1):
        t0 = time.perf_counter()
        result: PipelineResult = run_from_text(row["query"], mode=mode)
        wall = (time.perf_counter() - t0) * 1000

        core.append(result.core_ms)
        total.append(wall)
        for stage, ms in result.stage_timings_ms.items():
            stages.setdefault(stage, []).append(ms)
        for stage, ms in result.diagnostics.items():
            if stage.endswith("_ms"):
                stages.setdefault(stage, []).append(ms)

        if not result.within_budget:
            over_budget += 1
        if not result.reranked:
            rerank_skipped += 1
        if not result.allowed:
            refused["total"] += 1
            key = result.guardrail_stage or "unknown"
            refused[key] = refused.get(key, 0) + 1

        # Retrieval quality against the dataset's own labels. Scored on what
        # retrieval actually returned, independent of whether a guardrail
        # later decided not to answer.
        relevant = set(row["relevant_passage_ids"])
        got = [c.metadata.get("passage_id") for c in result.chunks]
        if not got and result.diagnostics:
            got = []
        recall_5.append(1.0 if relevant & set(got[:5]) else 0.0)
        recall_1.append(1.0 if relevant & set(got[:1]) else 0.0)
        mrr.append(_reciprocal_rank(got, relevant))

        records.append({
            "query_id": row["query_id"],
            "core_ms": result.core_ms,
            "allowed": result.allowed,
            "stage": result.guardrail_stage,
            "reranked": result.reranked,
        })
        if i % 20 == 0:
            print(f"  {i}/{len(sample)} ... core p50 so far {pct(core, 50)} ms")

    answered = [r for r in records if r["allowed"]]
    report = {
        "config": {
            "lang": config.LANG,
            "embed_model": config.EMBED_MODEL,
            "rerank_model": config.RERANK_MODEL,
            "chunk_strategy": config.ACTIVE_CHUNK_STRATEGY,
            "generation_mode": mode or config.GENERATION_MODE,
            "n_queries": len(sample),
            "core_budget_ms": config.CORE_BUDGET_MS,
            "torch_threads": config.TORCH_NUM_THREADS,
        },
        "core_latency_ms": summarize(core),
        "wall_latency_ms": summarize(total),
        "stages_ms": {k: summarize(v) for k, v in sorted(stages.items())},
        "budget": {
            "target_ms": config.CORE_BUDGET_MS,
            "within_budget": len(sample) - over_budget,
            "over_budget": over_budget,
            "pct_within": round(100 * (len(sample) - over_budget) / len(sample), 1),
            "rerank_skipped": rerank_skipped,
        },
        "retrieval_quality": {
            "recall_at_1": round(statistics.mean(recall_1), 4),
            "recall_at_5": round(statistics.mean(recall_5), 4),
            "mrr": round(statistics.mean(mrr), 4),
            "note": "scored on answered+refused alike; refusals return no chunks",
        },
        "guardrails": {
            "answered": len(answered),
            "refused": refused["total"],
            "refusal_rate": round(refused["total"] / len(sample), 4),
            "by_stage": {k: v for k, v in refused.items() if k != "total"},
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return report


def bench_stt(limit: int = None) -> dict:
    """Time the real Sarvam STT call against the committed sample audio."""
    from stt.sarvam_client import transcribe_with_retry

    audio_dir = config.BASE_DIR / "tests" / "sample_audio"
    files = sorted(audio_dir.glob("*.wav"))[: limit or 100]
    if not files:
        return {"error": "no sample audio found"}
    if not config.SARVAM_API_KEY:
        return {"skipped": "SARVAM_API_KEY not set"}

    timings, failures = [], 0
    for f in files:
        try:
            t0 = time.perf_counter()
            transcribe_with_retry(str(f), config.STT_LANGUAGE_CODE)
            timings.append((time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001
            print(f"  STT failed on {f.name}: {type(exc).__name__}")
            failures += 1
    return {"latency_ms": summarize(timings), "failures": failures,
            "n_files": len(files)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", default=None, help="extractive|llm")
    ap.add_argument("--with-stt", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = run(n=args.n, mode=args.mode)
    if args.with_stt:
        print("Benchmarking STT against sample audio ...")
        report["stt"] = bench_stt()

    out = args.out or config.BENCHMARK_RESULTS_PATH
    config.ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    c = report["core_latency_ms"]
    print("\n" + "=" * 62)
    print(f"  RAG core latency over {report['config']['n_queries']} real queries")
    print("=" * 62)
    print(f"  P50  {c['p50']:>8.1f} ms")
    print(f"  P70  {c['p70']:>8.1f} ms")
    print(f"  P90  {c['p90']:>8.1f} ms")
    print(f"  P100 {c['p100']:>8.1f} ms   (target {config.CORE_BUDGET_MS:.0f} ms)")
    print("-" * 62)
    b = report["budget"]
    print(f"  within budget: {b['within_budget']}/{report['config']['n_queries']} "
          f"({b['pct_within']}%)   rerank skipped: {b['rerank_skipped']}")
    q = report["retrieval_quality"]
    print(f"  Recall@1 {q['recall_at_1']:.3f}   Recall@5 {q['recall_at_5']:.3f}   "
          f"MRR {q['mrr']:.3f}")
    g = report["guardrails"]
    print(f"  refused: {g['refused']}/{report['config']['n_queries']} {g['by_stage']}")
    print("=" * 62)
    print(f"\nFull report -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
