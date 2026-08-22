# Latency report

*Generated 2026-08-22T21:05:14 by `python -m analytics.make_report`.*

## What is being measured

The **RAG core** — retrieval + generation + guardrails. This is the part of the system we control and the part the 200 ms target applies to.

Speech-to-text is **excluded from these percentiles and reported separately** below. Sarvam STT is an HTTPS round trip to a third-party API costing 613 ms at p50 and 978 ms at p100; folding someone else's network call into a "sub-200 ms" claim would make the number meaningless.

## Headline

| Percentile | RAG core latency | Target 200 ms |
|---|---|---|
| **P50** | **95.3 ms** | PASS |
| **P70** | **103.4 ms** | PASS |
| **P100** | **161.4 ms** | PASS |
| P90 | 124.3 ms | PASS |
| P95 | 127.4 ms | PASS |

**100/100 queries (100.0%) completed inside the 200 ms budget.**

## Methodology

- **100 real queries** sampled (seed 42) from the held-out eval split of `ai4bharat/IndicMSMARCO` — the dataset's own query→passage pairs, not hand-picked examples.
- Models are **warmed up before measuring**. A cold first query costs seconds of lazy init; including it would make P100 a measure of process startup.
- `p100` is the true worst observed case, not a trimmed maximum.
- Single process, `TORCH_NUM_THREADS=1`, generation mode `extractive`.
- Host: macOS-26.3-arm64-arm-64bit, Python 3.11.15.
- Measured on a **non-idle developer machine** (load average ~3.5). That is deliberate: the adaptive reranker is meant to hold the budget under contention, and these numbers show it doing so.

## Per-stage breakdown

| Stage | P50 | P70 | P90 | P100 |
|---|---|---|---|---|
| Query embedding | 20.11 ms | 24.10 ms | 36.34 ms | 52.77 ms |
| Dense search (FAISS) | 0.17 ms | 0.19 ms | 0.24 ms | 0.40 ms |
| Sparse search (BM25) | 2.63 ms | 3.25 ms | 5.00 ms | 7.13 ms |
| RRF fusion | 0.05 ms | 0.06 ms | 0.08 ms | 0.30 ms |
| Cross-encoder rerank | 70.48 ms | 77.41 ms | 87.62 ms | 108.59 ms |
| **Retrieval total** | 74.03 ms | 80.06 ms | 91.50 ms | 116.09 ms |
| Generation (extractive) | 0.34 ms | 0.38 ms | 0.48 ms | 0.75 ms |
| Guardrails (pre-retrieval) | 0.06 ms | 0.07 ms | 0.10 ms | 0.16 ms |
| Guardrails (post-generation) | 0.08 ms | 0.09 ms | 0.13 ms | 0.37 ms |
| **Core total** | **95.3 ms** | **103.4 ms** | **124.3 ms** | **161.4 ms** |

The cross-encoder rerank is **~74% of the median budget** — the single dominant cost, and the reason the pipeline reranks adaptively rather than at a fixed width.

## Adaptive reranking

- Reranker fully skipped on **6/100** queries where no budget headroom remained.
- On the rest it narrows the candidate list to what the remaining budget affords rather than dropping the stage outright.
- Effect of the change, same 100 queries, same machine:

| Rerank policy | P50 | P70 | P100 | Within budget |
|---|---|---|---|---|
| Fixed width (8 candidates) | 100.5 ms | 113.3 ms | 210.6 ms | 99% |
| **Adaptive width** | **95.3 ms** | **103.4 ms** | **161.4 ms** | **100.0%** |

## Retrieval quality

Scored against the dataset's own query→passage labels, on the same run as the latency figures.

| Metric | Value |
|---|---|
| Recall@1 | 0.690 |
| Recall@5 | 0.740 |
| MRR | 0.711 |

## Chunking strategies compared

`python -m analytics.compare_chunking`, 150 queries, chunk_size=400, overlap=80.

| Strategy | Chunks | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| semantic | 1467 | 0.747 | 0.860 | 0.796 |
| metadata | 1966 | 0.733 | 0.860 | 0.789 |
| recursive *(active)* | 1478 | 0.753 | 0.853 | 0.798 |
| fixed | 1469 | 0.653 | 0.807 | 0.709 |

Fixed-size chunking is clearly worst — it splits mid-sentence, so the embedding sees fragments of two ideas rather than one. That gap is the concrete justification for implementing more than one strategy. The top three are statistically tied at this sample size; `recursive` is active because it takes the best Recall@1 and MRR with a third fewer chunks than `metadata`.

## Guardrail activity

- Answered: **92/100**
- Refused: **8/100** (8.0%) — by stage: `{'low_confidence': 2, 'off_topic': 6}`

These are all in-domain dataset queries, so refusals here are **false refusals**. The rate matches the ~8.3% predicted by `analytics/calibrate_guardrails.py` at the chosen thresholds. It is a deliberate trade: the same operating point catches ~69% of genuinely off-topic queries. Both thresholds are env-tunable (`OFF_TOPIC_CENTROID_THRESHOLD`, `MIN_RELEVANCE_SCORE`).

## Refusals are cheap

Guardrails short-circuit before the expensive stages, so declining to answer costs a fraction of answering:

| Outcome | Typical core latency |
|---|---|
| Unsafe input (blocked pre-retrieval) | ~12 ms |
| Off-topic (blocked pre-retrieval) | ~24–36 ms |
| Full grounded answer | ~95 ms |

## Speech-to-text (reported separately)

Measured over 11 sample audio files (0 failures).

| Percentile | STT latency |
|---|---|
| P50 | 613 ms |
| P70 | 629 ms |
| P100 | 978 ms |

This is why the 200 ms target is scoped to the RAG core. STT latency is a property of a third-party API, not of this pipeline.

## Reproducing

```bash
python -m data.build_corpus          # fetch + flatten the dataset
python -m data.build_index           # chunk, embed, index
python -m analytics.run_benchmark --n 100
python -m analytics.compare_chunking --n 150
python -m analytics.calibrate_guardrails
python -m analytics.make_report
```
