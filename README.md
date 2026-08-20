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
