# Architecture

How the voice-enabled RAG pipeline is put together, and why each piece is the
way it is. Numbers quoted here come from `analytics/` and are reproducible with
the commands in each section.

---

## 1. Pipeline shape

```
audio ─► silence gate ─► STT (Sarvam) ─► pre-guardrails ─┬─► REFUSE (short-circuit)
                                                          │
                                                          ▼
                                    query embedding ─► hybrid retrieval
                                                          │
                                              dense (FAISS) + sparse (BM25)
                                                          │
                                                    RRF fusion
                                                          │
                                            cross-encoder rerank (adaptive)
                                                          │
                                                          ▼
                                                     generation
                                                          │
                                                 post-guardrails ─┬─► REFUSE
                                                                   │
                                                                   ▼
                                                          grounded answer
                                                          + citations
```

Every stage is timed individually. `harness/orchestrator.py` owns sequencing,
budget accounting, and the fallback paths; the stages themselves know nothing
about each other.

---

## 2. The 200 ms budget, stated precisely

**The 200 ms target covers the RAG core: retrieval + generation + guardrails.
Speech-to-text is measured and reported separately.**

This is not a dodge, it is the only honest way to state it. Sarvam STT is an
HTTPS round trip to a third-party API; measured against the committed sample
audio it costs **~1200–1700 ms**, and no amount of local engineering changes
that. Folding a network call to someone else's service into a "sub-200 ms
pipeline" claim would make the number meaningless.

So the split is explicit everywhere in the code:

- `PipelineResult.core_ms` — retrieval + generation + guardrails. Budget applies.
- `PipelineResult.stage_timings_ms["stt_ms"]` — reported, excluded from `core_ms`.
- `PipelineResult.within_budget` — whether *this run's* core made the target.

The text endpoint `/api/ask` exercises exactly the part we control, which is
why the benchmark drives it.

### Where the budget goes

Measured over 100 real dataset queries (see `analytics/latency_report.md`):

| Stage | Typical share |
|---|---|
| Query embedding | ~25 ms |
| Dense search (FAISS) | ~0.2 ms |
| BM25 search | ~3 ms |
| RRF fusion | ~0.1 ms |
| Cross-encoder rerank | ~70 ms — **the dominant cost** |
| Generation (extractive) | ~0.4 ms |
| Guardrails (both sides) | ~0.15 ms |

The reranker is ~70% of the budget. That single fact drove three decisions:
the adaptive rerank width, the extractive generation default, and reusing
retrieval's own scores for the off-topic guardrail instead of running a second
model.

### Two things that made the budget achievable

**Single-threaded inference.** Default torch thread pools contend badly on
one-query-at-a-time workloads. Measured on this machine:

| `TORCH_NUM_THREADS` | embed p50 | embed p100 | rerank(12) p50 |
|---|---|---|---|
| 1 | **10.5 ms** | **11.8 ms** | 70.7 ms |
| 2 | 16.0 ms | 48.6 ms | 82.8 ms |
| 4 | 11.3 ms | 17.0 ms | 39.9 ms |

Single-threaded wins on both median and, more importantly, on tail
predictability — which is what a percentile budget actually cares about.
`config.py` pins the OMP/MKL/BLAS env vars before torch is imported.

**Warm boot.** Both transformer models lazily allocate on their first forward
pass; a cold first query costs seconds. The server warms up during startup
(`retrieval.pipeline.warmup`) so a real user never pays that, and the
benchmark warms up before measuring so P100 measures the pipeline rather than
process startup.

---

## 3. Adaptive reranking — degrade, don't fail

The naive approach is a boolean: if we're running late, skip the reranker.
That throws away all retrieval quality the moment the machine hiccups.

Because the cross-encoder's cost is close to linear in candidate count, the
engine instead **spends exactly the candidates the remaining budget affords**:

```
remaining  = CORE_BUDGET_MS − elapsed − POST_RERANK_RESERVE_MS
affordable = remaining / measured_ms_per_candidate
n          = clamp(affordable, 0, RRF_TOP_N)
```

A slow query reranks 4 candidates instead of 8 and still gets reranked. Only a
query with no headroom left skips entirely. `RetrievalResult.reranked` and
`diagnostics["n_reranked"]` report what actually happened, so degradation is
always visible rather than silent.

`measured_ms_per_candidate` is an EWMA of real observations, so the pipeline
calibrates itself to whatever hardware it lands on instead of trusting a
constant tuned on a laptop. It is:

- **asymmetric** — rises slowly (α=0.2), falls quickly (α=0.5), so one CPU
  spike doesn't collapse rerank width while the machine has already recovered;
- **clamped** to [2, 20] ms, so the estimate can never run away in either
  direction. Without the ceiling there is a feedback trap: a spike inflates the
  estimate → fewer candidates → the estimate stays high forever.

---

## 4. Retrieval

### Why hybrid

Dense embeddings match meaning but lose exact tokens — numbers, names, units,
transliterated loanwords. In Hindi MS MARCO the answer often hinges on a
specific figure, which is exactly what BM25 catches and dense retrieval blurs.

The two score scales are incomparable (L2 distance vs term-frequency logits),
so they are merged with **Reciprocal Rank Fusion**, which uses only rank
position: `score = Σ 1/(k + rank)`. No tuning, no scale normalization.

The cross-encoder then rescores the shortlist by reading query and passage
*together*, rather than comparing two independently-computed vectors. It is far
more accurate and far more expensive, which is why it only ever sees the top-k.

### Unit vectors, exact index

Chunk vectors are L2-normalized at index time. With unit vectors, FAISS
`IndexFlatL2` distance and cosine similarity are monotonically related
(`cos = 1 − d²/2`), so an exact index yields calibrated cosine scores for free
— which is what the off-topic guardrail thresholds against. At 1,478 chunks an
exact index is also simply faster than an approximate one (~0.2 ms).

### Chunking — four strategies, scored

All four are implemented and measured against the dataset's own
query→passage labels (`python -m analytics.compare_chunking`, 150 queries):

| Strategy | Chunks | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| semantic | 1467 | 0.747 | **0.860** | 0.796 |
| metadata | 1966 | 0.733 | **0.860** | 0.789 |
| **recursive** (active) | 1478 | **0.753** | 0.853 | **0.798** |
| fixed | 1469 | 0.653 | 0.807 | 0.709 |

The real finding is that **fixed-size chunking is clearly worst** — it splits
mid-sentence, so the embedding sees fragments of two ideas instead of one.
That gap (0.653 vs 0.753 Recall@1) is well outside noise and is the concrete
justification for doing more than one naive split.

The top three are statistically tied — at n=150 the 0.007 Recall@5 gap between
semantic and recursive is one query. `recursive` is the default because it
takes the best Recall@1 and MRR, and produces a third fewer chunks than
`metadata` for the same Recall@5.

**A Hindi-specific bug worth recording:** the semantic and recursive splitters
originally split only on `[.!?]`. Devanagari ends sentences with the danda
(`।`, U+0964), so on Hindi text neither rule ever fired and every passage
collapsed into a single chunk. Both splitters are now danda-aware, and
`tests/test_chunking.py::test_hindi_actually_splits` locks that in.

---

## 5. Guardrails

Four gates, placed either side of the expensive work.

| Gate | When | Cost | What it catches |
|---|---|---|---|
| Silence | before STT | ~1 ms | empty/quiet audio, which STT otherwise returns as a confident hallucination |
| Unsafe input | before retrieval | ~0.1 ms | weapons, self-harm, violence, drug synthesis, CSAM, cyber-attack |
| Off-topic | before retrieval | ~0.05 ms | queries nowhere near the corpus |
| Low confidence | after retrieval | free | best passage too weak to answer from |
| Grounding | after generation | ~0.1 ms | answers not supported by the cited passages |

Refusing is **cheap and short-circuiting**: an unsafe query costs ~12 ms and an
off-topic one ~24–36 ms, versus ~128 ms for a full answer, because neither ever
reaches retrieval or generation.

### Off-topic detection is two-stage, and that's deliberate

**Stage 1, pre-retrieval:** compare the query vector against 24 k-means
centroids of the corpus (built at index time). One `(24, 384)` matmul —
microseconds — and it decides *before* we pay for retrieval, reranking, or
generation.

**Stage 2, post-retrieval:** the cross-encoder has now actually read the query
against the best passages. Its top score is a much better signal than any
pre-retrieval heuristic, and it costs nothing extra because retrieval already
computed it.

Two stages because they trade off differently: stage 1 is cheap but coarse,
stage 2 is accurate but only exists after the work is done.

### Thresholds are calibrated, not guessed

`analytics/calibrate_guardrails.py` scores 120 real dataset queries against 39
deliberately out-of-corpus ones and grid-searches the two-signal rule. Neither
signal separates the populations alone:

| Rule | False refusals | Off-topic caught |
|---|---|---|
| centroid < 0.33 only | 5.0% | 64.1% |
| rerank < −3.97 only | 5.8% | 12.8% |
| **centroid < 0.34 OR rerank < −4.5** | **8.3%** | **69.2%** |

The chosen operating point is the last row. It is an honest tradeoff, not a
solved problem: roughly 1 in 12 genuine questions gets refused, and roughly 3
in 10 off-topic ones still get through to the grounding check. Both thresholds
are env-tunable.

### Grounding

Takes the answer's content tokens (stopwords stripped), and measures what
fraction appear in the passages actually cited. Below
`GROUNDING_OVERLAP_THRESHOLD` (0.55) the answer is refused.

An NLI/entailment cross-encoder would be more accurate, but it is another
40–90 ms on a 200 ms budget — it would become the second-largest cost in the
pipeline. Lexical overlap is a weaker signal, but it is the *right* weak
signal here: extractive answers are verbatim spans, and the failure mode that
matters (fabricated numbers, names, claims) is exactly what token overlap
catches.

Critically, this check runs **independently of what the generator claims**.
`tests/test_guardrails.py::test_post_check_refuses_ungrounded_generation`
asserts that a generator returning `grounded=True` on an invented answer is
still overruled.

---

## 6. Generation — two paths

**`extractive` (default).** Selects the best-matching sentences from the
retrieved chunks and returns them verbatim, scoring each sentence by
query-term coverage plus small position and chunk-rank priors.

- ~0.4 ms, so the pipeline fits the budget.
- **Grounded by construction** — the answer is a literal span of a retrieved
  passage and cannot hallucinate.
- Costs fluency: it returns the corpus's own words, not a synthesized reply.

**`llm`.** Calls Claude with a strict context-only prompt for a fluent answer.
Several hundred ms minimum, so it deliberately does not claim the budget.
Grounding is defended three ways — the context-only prompt, a parsed citation
back to the passage, and the same independent overlap check, because a prompt
is not a security boundary.

If the LLM path fails for any reason it **falls back to extractive** and
records the downgrade in `GenerationResult.mode`. A generation outage degrades
answer fluency; it does not take the pipeline down.

---

## 7. Harness responsibilities

`harness/orchestrator.py` owns what individual stages deliberately do not:

- **Structured I/O** — every stage boundary is a Pydantic model in
  `harness/schemas.py`. Fields added after day 1 are optional with defaults, so
  the original stub contract still works (`harness/stubs.py`, and the legacy
  `run_pipeline(audio, retrieve_fn, generate_fn, guardrail_fn)` entry point).
- **Budget accounting** — per-stage timing, `core_ms`, `within_budget`.
- **Short-circuiting** — a pre-retrieval refusal returns immediately.
- **Degradation over failure** — adaptive rerank on the retrieval side,
  extractive fallback on the generation side.
- **Retries** — exponential backoff on STT, but *only* for transient failures.
  A 401 fails immediately, because retrying bad credentials three times just
  triples the user's wait.
- **Never 500 on a logging failure** — `log_run` swallows its own exceptions.

---

## 8. Known limitations

Stated plainly, because a guardrails project that overclaims its guardrails
would be self-defeating:

1. **~31% of off-topic queries still reach generation.** The grounding check
   catches some but not all. A query about Goa weather can match a passage
   about Copenhagen weather closely enough to pass both gates.
2. **The unsafe filter is lexical.** It is a fast, transparent first line, not
   a complete moderation system. It will miss paraphrases and obfuscation.
3. **Grounding is token overlap, not entailment.** It catches fabricated
   specifics; it will not catch a fluent answer that reuses the passage's
   vocabulary while inverting its meaning.
4. **Corpus is 1,000 passages** (`ai4bharat/IndicMSMARCO` Hindi). The full
   `MSMARCO-XI` Hindi split is 3.7 GB; `MAX_CORPUS_ROWS` and `DATASET_REPO`
   scale it up, but retrieval latency will grow with the index.
5. **Latency figures are single-machine.** The adaptive reranker is the
   defence against slower hardware, but the absolute numbers will differ.
