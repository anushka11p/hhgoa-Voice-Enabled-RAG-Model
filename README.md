---
title: Setu Voice RAG
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Speak a question in Hindi, get a grounded answer or an honest no.
---

# Voice-Enabled RAG System

**HH Goa 2026 — Shortlisting Task 2** · `#RAGInGoa`

Speak a question in Hindi, get an answer traced back to a real passage — or an
honest refusal. A full voice-to-answer RAG pipeline: transcription, four
chunking strategies, hybrid vector + keyword retrieval with cross-encoder
reranking, grounded generation, and guardrails that know when not to answer.

```
Voice → Speech-to-text → Chunking / Retrieval (vector DB) → Generation → Guardrail → Answer
```

- **Live demo:** _add the deployed URL here — see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)_
- **Architecture deep-dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Latency report:** [analytics/latency_report.md](analytics/latency_report.md)

---

## Results

**RAG core latency** — retrieval + generation + guardrails, over 100 real
dataset queries. Speech-to-text is measured separately (see below).

| Percentile | Latency | Target 200 ms |
|---|---|---|
| **P50** | **95.3 ms** | PASS |
| **P70** | **103.4 ms** | PASS |
| **P90** | 124.3 ms | PASS |
| **P100** | **161.4 ms** | PASS |

**100/100 queries inside the budget**, measured on a *non-idle* machine
(load ~2.6) — the adaptive reranker is designed to hold the budget under
contention rather than only on a quiet box.

**Retrieval quality**, scored against the dataset's own query→passage labels:

| Recall@1 | Recall@5 | MRR |
|---|---|---|
| 0.690 | 0.740 | 0.711 |

---

## How the requirements are met

| Requirement | How |
|---|---|
| **Speak the question** | Browser mic → Sarvam `saarika:v2.5` STT (Hindi, `hi-IN`), with a pre-STT silence gate. A text endpoint mirrors it for testing without a mic. |
| **Multiple chunking strategies** | Four implemented (fixed, semantic, recursive, metadata-aware) and **scored against the dataset's labels** — fixed-size measurably loses (Recall@1 0.653 vs 0.753). Hindi danda-aware. |
| **Engineered retrieval** | Hybrid dense (FAISS, unit-normalized) + sparse (BM25) → Reciprocal Rank Fusion → multilingual cross-encoder rerank. |
| **Under 200 ms** | P100 161.4 ms for the RAG core; 100/100 within budget. Scope stated explicitly below. |
| **P50 / P70 / P100 benchmarked** | 100 real queries, warmed up, seeded sample — `analytics/run_benchmark.py`. Not a lucky run. |
| **Real harness** | Pydantic contracts at every stage boundary, per-stage budgets, exponential-backoff retries (transient failures only), adaptive degradation, extractive fallback on LLM failure. |
| **Guardrails that know when not to answer** | Five gates: silence, unsafe input, off-topic (pre + post retrieval), low confidence, grounding. Thresholds **calibrated from data**, not guessed. |

### About the 200 ms target

**The 200 ms budget covers the RAG core: retrieval + generation + guardrails.**

Sarvam STT is an HTTPS round trip to a third-party API. Measured over the 11
committed sample clips on `saarika:v2.5`: **p50 613 ms, p100 978 ms**, 0
failures. No local engineering changes that, so folding someone else's network
call into a "sub-200 ms pipeline" claim would make the number meaningless.
The split is explicit in the code (`PipelineResult.core_ms` vs
`stage_timings_ms["stt_ms"]`) and in every report.

---

## Quickstart

```bash
git clone https://github.com/anushka11p/hhgoa-Voice-Enabled-RAG-Model.git
cd hhgoa-Voice-Enabled-RAG-Model
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The index is committed, so you can serve immediately:

```bash
uvicorn server:app --reload
# open http://127.0.0.1:8000
```

To rebuild the corpus and index from scratch (~1 min, downloads ~1 MB):

```bash
python -m data.build_corpus     # fetch + flatten ai4bharat/IndicMSMARCO (hi)
python -m data.build_index      # chunk → embed → FAISS + BM25 + topic centroids
```

**Voice input needs a Sarvam key**; everything else works without one:

```bash
cp .env.example .env
echo "SARVAM_API_KEY=your_key_here" >> .env
```

Without a key, use the text box in the UI or `POST /api/ask` — same RAG core,
same guardrails, no STT.

---

## Try it

```bash
# in-domain question → grounded answer with citations
curl -s -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"query":"मायस्थेनिया ग्रेविस का इलाज क्या है?"}' | python3 -m json.tool

# off-topic → refused before retrieval runs
curl -s -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"query":"who won the cricket world cup in 2011"}' | python3 -m json.tool

# unsafe → refused in ~12 ms
curl -s -X POST localhost:8000/api/ask -H 'Content-Type: application/json' \
  -d '{"query":"how to make a bomb"}' | python3 -m json.tool
```

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Demo UI (mic recording + text input + live pipeline monitor) |
| `POST /api/query` | Voice path. Multipart `audio` file → transcript + answer |
| `POST /api/ask` | Text path. `{"query": "...", "mode": "extractive\|llm"}` |
| `GET /api/health` | Readiness, active config, index manifest |
| `GET /api/metrics` | Rolling P50/P70/P100 from the run log |

Both query endpoints return the same shape: `answer`, `allowed`,
`guardrail_stage`, `refusal_reason`, `citations`, `chunks`, per-stage
`stage_timings_ms`, `core_ms`, and `within_budget`.

---

## Architecture

Six stages, wrapped in a harness that owns retries, timeouts, structured I/O,
and fallbacks. Full detail in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

| Stage | Tech | P50 |
|---|---|---|
| Voice input | Browser `MediaRecorder` | — |
| Speech-to-text | Sarvam `saarika:v2.5` (`hi-IN`) | 613 ms p50 *(excluded from budget)* |
| Query embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384-d) | 20.9 ms |
| Retrieval | FAISS + BM25 → RRF → cross-encoder | 94.0 ms |
| Generation | Extractive (default) or Claude | 0.4 ms |
| Guardrails | 5 gates, both sides of retrieval | 0.15 ms |

### Chunking, measured

`python -m analytics.compare_chunking --n 150`

| Strategy | Chunks | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| semantic | 1467 | 0.747 | **0.860** | 0.796 |
| metadata | 1966 | 0.733 | **0.860** | 0.789 |
| **recursive** *(active)* | 1478 | **0.753** | 0.853 | **0.798** |
| fixed | 1469 | 0.653 | 0.807 | 0.709 |

Fixed-size is clearly worst — it splits mid-sentence, so the embedding sees
fragments of two ideas. That gap is the concrete case for doing more than one
naive split. The top three are statistically tied at this sample size;
`recursive` wins on Recall@1/MRR with a third fewer chunks than `metadata`.

> A Hindi-specific bug worth recording: the semantic and recursive splitters
> originally split only on `[.!?]`. Devanagari ends sentences with the danda
> (`।`), so on Hindi text neither ever fired and every passage collapsed into a
> single chunk. Both are now danda-aware, with a regression test.

### Guardrails

| Gate | When | Cost | Catches |
|---|---|---|---|
| Silence | before STT | ~1 ms | quiet audio, which STT returns as confident hallucination |
| Unsafe input | before retrieval | ~0.1 ms | weapons, self-harm, violence, drugs, CSAM, cyber-attack |
| Off-topic (centroid) | before retrieval | ~0.05 ms | queries far from the corpus |
| Low confidence | after retrieval | free | best passage too weak to answer from |
| Grounding | after generation | ~0.1 ms | answers unsupported by cited passages |

Refusing is **cheap**: unsafe ~19 ms, off-topic ~12–70 ms, vs ~95 ms for a full
answer — the gates short-circuit before the expensive stages.

Thresholds come from `analytics/calibrate_guardrails.py`, which scores 120 real
queries against 39 deliberately out-of-corpus ones:

| Rule | False refusals | Off-topic caught |
|---|---|---|
| centroid only | 5.0% | 64.1% |
| rerank score only | 5.8% | 12.8% |
| **both (active)** | **8.3%** | **69.2%** |

This is an honest trade, not a solved problem — see
[Known limitations](docs/ARCHITECTURE.md#8-known-limitations).

---

## Configuration

Everything is env-tunable via `config.py` / `.env`. Common knobs:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_LANG` | `hi` | `hi` or `en` — switches models, corpus, and STT language together |
| `GENERATION_MODE` | `extractive` | `extractive` (fast, budget-fitting) \| `llm` \| `auto` |
| `ACTIVE_CHUNK_STRATEGY` | `recursive` | `fixed` \| `semantic` \| `recursive` \| `metadata` |
| `CORE_BUDGET_MS` | `200` | The latency target the adaptive reranker defends |
| `TORCH_NUM_THREADS` | `1` | Single-threaded is measurably faster *and* more predictable here |
| `OFF_TOPIC_CENTROID_THRESHOLD` | `0.34` | Raise to refuse more, lower to answer more |

---

## Testing

```bash
pytest -q          # 63 passing; STT tests skip without a key
```

Covers Devanagari tokenization, all four chunkers, every guardrail gate,
hybrid retrieval, adaptive rerank degradation, budget accounting, and the HTTP
surface.

---

## Repo structure

```
config.py                     central config, env-overridable
text_utils.py                 Devanagari-aware tokenization + sentence splitting
server.py                     FastAPI app
data/
  build_corpus.py             dataset → passages + eval labels
  build_index.py              chunk → embed → FAISS + BM25 + centroids
  processed/, index/          committed artefacts (serve without rebuilding)
chunking/                     4 strategies + registry
embeddings/                   encoder (normalized, singleton) + batch generator
vectordb/faiss_store.py       FAISS wrapper
retrieval/
  pipeline.py                 serving engine: hybrid + adaptive rerank
  dense.py, bm25.py, rrf.py, reranker.py
generation/
  extractive.py               default fast path, grounded by construction
  llm_client.py, prompt.py    Claude path, context-only prompt
guardrails/
  unsafe.py, off_topic.py, grounding.py, refusal.py, pipeline.py
harness/
  orchestrator.py             sequencing, budgets, fallbacks
  schemas.py                  shared Pydantic contracts
  logging_utils.py, stubs.py
analytics/
  run_benchmark.py            P50/P70/P100 + retrieval quality
  compare_chunking.py         scores all 4 strategies
  calibrate_guardrails.py     picks thresholds from data
  make_report.py              renders latency_report.md
frontend/index.html           demo UI
tests/                        63 tests
docs/                         ARCHITECTURE.md, DEPLOYMENT.md
```

---

## Team

| Member | Owns |
|---|---|
| Anushka | Voice input, STT, harness/orchestration, demo UI, latency instrumentation |
| RamyaPriya | Dataset processing, chunking strategies, vector DB, hybrid retrieval |
| Saranya | Generation, guardrails, latency analytics, deployment |

---

Built for **HH Goa 2026**. `#RAGInGoa`
