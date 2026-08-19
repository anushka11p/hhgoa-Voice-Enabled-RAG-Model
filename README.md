# Voice-Enabled RAG System

**HH Goa 2026 — Shortlisting Task 2**

A voice-enabled Retrieval-Augmented Generation pipeline: a user speaks a question, the system transcribes it, retrieves relevant context from the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset, and returns a grounded answer — end to end, under a 200ms latency budget, orchestrated through a proper harness with guardrails.

```
Voice input → Speech-to-text → Chunking / Retrieval (vector DB) → Answer generation → Guardrail check → Final answer
```

**Live demo:** 
**Demo video:** 
**Process video:** 

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Chunking strategy](#chunking-strategy)
- [Retrieval](#retrieval)
- [Guardrails](#guardrails)
- [Latency results](#latency-results)
- [Repo structure](#repo-structure)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Team](#team)

---

## Overview

| Requirement | How we meet it |
|---|---|
| Speech-to-text | [Sarvam / ElevenLabs — pick one] |
| Chunking | 4 strategies: fixed-size, semantic, recursive, metadata-aware |
| Latency | Full pipeline (chunk → retrieve → generate) targets < 200ms |
| Latency analytics | P50 / P70 / P100 measured across [N] test queries |
| Harness | Structured orchestrator with retries, timeouts, structured I/O, fallback paths |
| Guardrails | Off-topic filtering, unsafe-input filtering, grounding/hallucination checks, confidence-based refusal |

## Architecture

| Stage | What it does | Key tech |
|---|---|---|
| 1. Voice input | Captures mic/audio input from the user | Browser mic / uploaded audio file |
| 2. Speech-to-text | Converts audio to a clean transcript | Sarvam or ElevenLabs Scribe |
| 3. Query embedding | Embeds the cleaned transcript for retrieval | sentence-transformers |
| 4. Retrieval | Hybrid dense + sparse search across the chunk index, then reranked | Qdrant/Chroma + BM25 + cross-encoder reranker |
| 5. Generation | Produces an answer grounded strictly in retrieved chunks | Claude/GPT API, context-only prompt |
| 6. Guardrail check | Filters off-topic/unsafe queries, checks grounding, decides whether to answer or refuse | Similarity thresholds, NLI/entailment check |

All stages run inside `harness/orchestrator.py`, which owns retries, per-stage timeouts, structured Pydantic I/O, logging, and fallback paths.

### Interface contracts

```
STT output       -> { transcript: str, language: str, confidence: float }
Retrieval input   -> { query: str, top_k: int, filters: dict }
Retrieval output  -> { chunks: [ {text, score, metadata} ], retrieval_ms: float }
Generation input  -> { query: str, chunks: [...] }
Generation output -> { answer: str, grounded: bool, citations: [chunk_id] }
Guardrail output  -> { allowed: bool, reason: str|None, final_answer: str }
```

These shapes are defined once in `harness/schemas.py` and imported by every module — that's the contract that keeps the STT, retrieval, and generation code compatible.

## Dataset

We use [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — MS MARCO translated into Indic languages. Each example contains a query, an `answers` field, and a list of candidate passages, plus `source_lang` / `target_lang` / translation metadata.

```python
from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
```

For fast local iteration we prototype against the smaller [`ai4bharat/IndicMSMARCO`](https://huggingface.co/datasets/ai4bharat/IndicMSMARCO) (~1,000 rows/language) before running the final build against the full dataset.

Because every row pairs a query with its relevant passage(s), the dataset also serves as our retrieval-quality eval set and our latency-benchmark query set.

## Chunking strategy

We deliberately use more than one chunking approach:

1. **Fixed-size with overlap** — baseline, ~256 tokens, ~20% overlap
2. **Semantic chunking** — splits on embedding-similarity breakpoints so chunks stay topically coherent
3. **Recursive / sentence-aware chunking** — language-aware separators, falling back to fixed-size only when a paragraph is too long
4. **Metadata-aware chunking** — every chunk tagged with `doc_id`, `passage_id`, `language`, `source_query` for filtered/boosted retrieval

See [`data/chunking/`](./data/chunking) for implementations and rationale.

## Retrieval

Hybrid retrieval combining:
- Dense vector search (embeddings, cosine similarity)
- Sparse BM25 keyword search
- Reciprocal rank fusion to combine both result sets
- Cross-encoder reranking on the fused top-k before it reaches the LLM

## Guardrails

- **Off-topic detection** — query is checked against the corpus domain before retrieval runs
- **Unsafe-input filtering** — basic content moderation on the transcribed query
- **Grounding check** — after generation, verify the answer's claims are supported by the retrieved chunks
- **Confidence-threshold refusal** — if retrieval similarity is below threshold, the system responds "I don't have enough information" instead of guessing

## Latency results

Measured across **[N] end-to-end test queries** (pipeline: chunk retrieval → generation → guardrail check).

| Percentile | Latency (ms) |
|---|---|
| P50 | [ ] |
| P70 | [ ] |
| P100 | [ ] |

Full breakdown per stage and methodology: [`analytics/latency_report.md`](./analytics/latency_report.md)

## Repo structure

```
voice-rag/
├── README.md
├── requirements.txt
├── .env.example
├── data/                  # dataset loading + chunking
│   ├── load_dataset.py
│   └── chunking/
│       ├── fixed_size.py
│       ├── semantic.py
│       ├── recursive.py
│       └── metadata_aware.py
├── retrieval/             # vector store + hybrid search
│   ├── vector_store.py
│   ├── bm25.py
│   └── reranker.py
├── stt/                   # speech-to-text
│   └── sarvam_client.py
├── harness/                # orchestration + shared schemas
│   ├── orchestrator.py
│   ├── schemas.py
│   └── logging_utils.py
├── generation/             # answer generation
│   └── llm_client.py
├── guardrails/             # safety + grounding checks
│   ├── off_topic.py
│   ├── grounding_check.py
│   └── refusal.py
├── analytics/               # latency benchmarking
│   ├── run_benchmark.py
│   └── latency_report.md
├── app.py                  # demo UI
└── tests/
```

## Setup

```bash
git clone https://github.com/<org>/voice-rag.git
cd voice-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY / ELEVENLABS_API_KEY, LLM API key, vector DB config
```

## Running the pipeline

```bash
# 1. Build the index (one-time)
python data/load_dataset.py
python retrieval/vector_store.py --build-index

# 2. Run the demo app
python app.py

# 3. Run the latency benchmark
python analytics/run_benchmark.py --queries 100
```

## Team

| Member | Owns |
|---|---|
| Anushka | Voice input, STT, harness/orchestration, demo UI, latency instrumentation |
| RamyaPriya | Dataset processing, chunking strategies, vector DB, hybrid retrieval |
| Saranya | Generation, guardrails, latency analytics, README/deployment |

---

Built for **HH Goa 2026**. `#RAGInGoa`
