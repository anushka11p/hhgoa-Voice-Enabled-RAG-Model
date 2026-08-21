# Voice RAG Project

A Voice-based Retrieval-Augmented Generation (RAG) system built as part of a team project. This module specifically covers:
- Data processing and dataset exploration
- Document chunking (Fixed, Recursive, Semantic, Metadata)
- Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Fast vector similarity search using `faiss-cpu`
- Hybrid retrieval (Dense Retrieval, BM25, Reciprocal Rank Fusion, MS MARCO Reranker)
- Retrieval Evaluation (Recall@5, Recall@10, MRR)

## Project Structure
- `data/` - Holds raw and processed dataset files.
- `notebooks/` - Contains Jupyter notebooks for dataset exploration.
- `chunking/` - Modular chunking strategies.
- `embeddings/` - Embedding logic.
- `vectordb/` - FAISS integration.
- `retrieval/` - Hybrid retrieval pipeline.
- `evaluation/` - Evaluation metrics.

## Getting Started
1. Create a Python 3.11 virtual environment.
2. Install dependencies via `pip install -r requirements.txt`.
# Voice-Enabled RAG System

**HH Goa 2026 — Shortlisting Task 2**

A voice-enabled Retrieval-Augmented Generation pipeline: a user speaks a question, the system transcribes it, retrieves relevant context from the [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) dataset, and returns a grounded answer — end to end, orchestrated through a proper harness with guardrails.

```
Voice input → Speech-to-text → Chunking / Retrieval (vector DB) → Answer generation → Guardrail check → Final answer
```

**Live demo:** [add Render link here once deployed]
**Demo video:** [add link here]
**Process video:** [add link here]

---

## Table of contents

- [Overview](#overview)
- [Current status](#current-status)
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
| Speech-to-text | Sarvam (`saarika:v2`), Hindi (`hi-IN`) |
| Chunking | 4 strategies: fixed-size, semantic, recursive, metadata-aware *(in progress)* |
| Latency | Full pipeline targets a low end-to-end latency; measured and logged per stage |
| Latency analytics | P50 / P70 / P100 to be computed across a real test query set |
| Harness | Structured FastAPI-based orchestrator with retries, structured I/O, error recovery, and a pre-STT silence guardrail |
| Guardrails | Off-topic handling, unsafe-input handling, and grounding checks *(in progress — currently stubbed)* |

## Current status

This project is under active development. Here's what's real vs. placeholder right now:

**Working and tested:**
- Speech-to-text (Sarvam) — transcribes real Hindi audio with retry logic
- Orchestrator/harness — runs the full pipeline shape end to end with per-stage timing and structured error handling
- Pre-STT silence detection — rejects near-silent audio before it reaches the STT API (calibration ongoing)
- Demo UI — FastAPI backend + themed frontend with live mic recording
- Latency logging — every pipeline run is logged to `analytics/latency_log.jsonl`

**Still placeholder (stub functions):**
- Retrieval — currently returns a hardcoded example chunk, not real dataset search
- Generation — currently echoes the query back as a stub answer, not a real LLM answer
- Guardrails (off-topic/unsafe/grounding) — currently always returns "allowed," no real filtering yet

The pipeline architecture, timing, and UI are fully built and wired correctly — once real retrieval, generation, and guardrail modules are dropped in, no structural changes should be needed, only swapping three function imports in `server.py`.

## Architecture

| Stage | What it does | Key tech |
|---|---|---|
| 1. Voice input | Captures mic/audio input from the user | Browser mic (`getUserMedia`/`MediaRecorder`) |
| 2. Speech-to-text | Converts audio to a clean transcript | Sarvam `saarika:v2`, Hindi |
| 3. Query embedding | Embeds the cleaned transcript for retrieval | *(pending — retrieval module in progress)* |
| 4. Retrieval | Hybrid dense + sparse search across the chunk index, then reranked | *(pending — retrieval module in progress)* |
| 5. Generation | Produces an answer grounded strictly in retrieved chunks | *(pending — generation module in progress)* |
| 6. Guardrail check | Filters off-topic/unsafe queries, checks grounding, decides whether to answer or refuse | *(pending — guardrail module in progress)* |

All stages run through `harness/orchestrator.py`, which owns retries, structured Pydantic I/O, per-stage timing, and error handling. The FastAPI backend (`server.py`) exposes this as a single `/api/query` endpoint that accepts an audio file and returns transcript, answer, timings, and an `allowed` flag.

### Interface contracts

```
STT output        -> { transcript: str, language: str, confidence: float, duration_ms: float }
Retrieval output   -> { chunks: [ {text, score, metadata} ], retrieval_ms: float }
Generation output  -> { answer: str, grounded: bool, citations: [chunk_id] }
Guardrail output   -> { allowed: bool, reason: str|None, final_answer: str }
```

These shapes are defined once in `harness/schemas.py` and shared across every module — this is the contract that keeps STT, retrieval, generation, and guardrail code compatible regardless of who builds what.

## Dataset

We use [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — MS MARCO translated into Indic languages. The project is built end-to-end in **Hindi**: STT transcribes Hindi speech, and retrieval/generation will operate against the Hindi (`hi`) config of the dataset.

```python
from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI", "hi", split="train")
```

Because every row pairs a query with its relevant passage(s), the dataset also serves as the retrieval-quality eval set and the latency-benchmark query set once retrieval is wired in.

## Chunking strategy

*(In progress.)* Planned approach uses more than one chunking strategy rather than a single fixed-size split:

1. **Fixed-size with overlap** — baseline
2. **Semantic chunking** — splits on embedding-similarity breakpoints
3. **Recursive / sentence-aware chunking** — Hindi sentence-boundary aware
4. **Metadata-aware chunking** — tagged with `doc_id`, `passage_id`, `language`, `source_query`

Retrieval is planned to combine dense vector search with BM25 sparse search via reciprocal rank fusion, then rerank with a cross-encoder.

## Retrieval

*(In progress — currently a stub.)* Will combine dense embedding search and BM25 keyword search over the chunked Hindi corpus, fused and reranked before reaching generation.

## Guardrails

Currently implemented:
- **Pre-STT silence detection** — audio is checked for near-silence before being sent to STT, since empty/quiet audio was found to sometimes produce hallucinated transcripts rather than an empty result. Calibration of the detection threshold is ongoing.

Planned, not yet implemented:
- **Off-topic detection** — reject queries unrelated to the corpus before retrieval runs
- **Unsafe-input filtering** — basic content moderation on the transcribed query
- **Grounding check** — verify generated answers are actually supported by retrieved passages
- **Confidence-threshold refusal** — respond with "I don't have enough information" when retrieval confidence is low

## Latency results

*(Pending — full benchmark requires real retrieval and generation to be meaningful.)* Per-stage timings are already being logged for every pipeline run in `analytics/latency_log.jsonl`, including stub-stage runs, so the benchmarking script can be run as soon as real modules are in place.

| Percentile | Latency (ms) |
|---|---|
| P50 | *pending* |
| P70 | *pending* |
| P100 | *pending* |

## Repo structure

```
hhgoa-Voice-Enabled-RAG-Model/
├── README.md
├── requirements.txt
├── .env                    # not committed — holds SARVAM_API_KEY
├── server.py                # FastAPI backend, single /api/query endpoint
├── frontend/
│   └── index.html            # themed demo UI, mic recording + live results
├── stt/
│   ├── sarvam_client.py       # Sarvam STT integration + retry logic
│   └── audio_check.py          # pre-STT silence detection
├── harness/
│   ├── orchestrator.py          # pipeline sequencing, timing, error handling
│   ├── schemas.py                 # shared Pydantic models
│   ├── stubs.py                     # placeholder retrieval/generation/guardrail functions
│   └── logging_utils.py               # writes per-run latency logs
├── analytics/
│   └── latency_log.jsonl                # per-run timing log
└── tests/
    ├── test_stt.py
    ├── test_orchestrator.py
    └── sample_audio/
```

*(`retrieval/`, `generation/`, and `guardrails/` folders will be added as those modules land.)*

## Setup

```bash
git clone https://github.com/anushka11p/hhgoa-Voice-Enabled-RAG-Model.git
cd hhgoa-Voice-Enabled-RAG-Model
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "SARVAM_API_KEY=your_key_here" > .env
```

## Running the pipeline

```bash
# Run the backend + demo UI
uvicorn server:app --reload
# then open http://127.0.0.1:8000
```

```bash
# Run individual tests
python tests/test_stt.py
python tests/test_orchestrator.py
```

## Team

| Member | Owns |
|---|---|
| Anushka | Voice input, STT, harness/orchestration, demo UI, latency instrumentation |
| RamyaPriya | Dataset processing, chunking strategies, vector DB, hybrid retrieval |
| Saranya | Generation, guardrails, latency analytics, deployment |

---

Built for **HH Goa 2026**. `#RAGInGoa`
