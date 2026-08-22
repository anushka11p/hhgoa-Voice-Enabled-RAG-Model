"""Stage 1 - chunk, embed, and persist the searchable index.

Produces a self-contained bundle in data/index/ so the server boots by loading
files rather than re-embedding 1k passages on every cold start:

  voice_rag.bin        FAISS IndexFlatL2 over unit-normalized chunk vectors
  voice_rag_meta.json  chunk metadata, row-aligned with the FAISS index
  voice_rag_bm25.pkl   fitted BM25Okapi model + its tokenized corpus
  voice_rag_topic.npz  k-means centroids of the corpus, for the off-topic gate

Run:  python -m data.build_index            (uses ACTIVE_CHUNK_STRATEGY)
      python -m data.build_index --strategy semantic
"""
import argparse
import json
import pickle
import sys
import time

import numpy as np

import config
from chunking.registry import chunk_documents, chunk_stats
from embeddings.encoder import encode_texts


def _topic_centroids(vectors: np.ndarray, n_centroids: int = 24, iters: int = 25) -> np.ndarray:
    """Summarize the corpus as a handful of unit-norm centroids.

    The off-topic guardrail needs to answer "is this query anywhere near what
    the corpus talks about?" before retrieval runs. Comparing against ~24
    centroids is a single (24, 384) matmul -- microseconds -- versus scanning
    every chunk.

    Implemented in plain numpy rather than faiss.Kmeans: the faiss-cpu build
    here segfaults against numpy 2.x, and spherical k-means over a few
    thousand unit vectors is a dozen lines and runs in well under a second.
    Vectors arrive unit-normalized, so cosine similarity is just a dot product
    and the mean of a cluster re-normalized is the spherical centroid.
    """
    n_centroids = max(1, min(n_centroids, len(vectors) // 8 or 1))
    rng = np.random.default_rng(42)

    # k-means++ style seeding: spread the initial centroids out.
    cents = [vectors[rng.integers(len(vectors))]]
    for _ in range(n_centroids - 1):
        sims = np.max(np.stack([vectors @ c for c in cents]), axis=0)
        far = np.clip(1.0 - sims, 1e-9, None)
        cents.append(vectors[rng.choice(len(vectors), p=far / far.sum())])
    cents = np.stack(cents).astype(np.float32)

    for _ in range(iters):
        assign = np.argmax(vectors @ cents.T, axis=1)
        moved = False
        for k in range(len(cents)):
            members = vectors[assign == k]
            if len(members) == 0:
                continue
            new_c = members.mean(axis=0)
            new_c /= max(float(np.linalg.norm(new_c)), 1e-9)
            if not np.allclose(new_c, cents[k], atol=1e-6):
                moved = True
            cents[k] = new_c
        if not moved:
            break

    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    return (cents / np.clip(norms, 1e-9, None)).astype(np.float32)


def build(strategy: str = None, index_name: str = None) -> dict:
    from rank_bm25 import BM25Okapi

    from vectordb.faiss_store import FaissStore

    strategy = strategy or config.ACTIVE_CHUNK_STRATEGY
    index_name = index_name or config.INDEX_NAME

    if not config.CORPUS_PATH.exists():
        raise SystemExit(
            f"Corpus not found at {config.CORPUS_PATH}. "
            "Run `python -m data.build_corpus` first."
        )
    docs = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(docs)} passages.")

    t0 = time.perf_counter()
    chunks = chunk_documents(docs, strategy=strategy)
    print(f"Chunked with '{strategy}': {chunk_stats(chunks)} "
          f"({(time.perf_counter() - t0) * 1000:.0f} ms)")

    texts = [c["chunk_text"] for c in chunks]
    t0 = time.perf_counter()
    vectors = encode_texts(texts, show_progress=True)
    print(f"Embedded {len(texts)} chunks in {time.perf_counter() - t0:.1f}s "
          f"-> {vectors.shape}")

    # --- FAISS ---
    store = FaissStore(vector_dim=config.VECTOR_DIM)
    store.index.add(vectors)
    store.metadata = chunks
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_index(str(config.INDEX_DIR), index_name)

    # --- BM25 ---
    # Devanagari has no casing, but lowercasing is still right for the Latin
    # tokens (units, brand names, loanwords) that pepper the Hindi passages.
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_path = config.INDEX_DIR / f"{index_name}_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump({"model": bm25, "tokenized": tokenized}, f)
    print(f"BM25 index -> {bm25_path}")

    # --- topic centroids for the off-topic guardrail ---
    cents = _topic_centroids(vectors)
    topic_path = config.INDEX_DIR / f"{index_name}_topic.npz"
    np.savez_compressed(topic_path, centroids=cents)
    print(f"Topic centroids -> {topic_path} {cents.shape}")

    manifest = {
        "strategy": strategy,
        "n_passages": len(docs),
        "n_chunks": len(chunks),
        "embed_model": config.EMBED_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "vector_dim": config.VECTOR_DIM,
        "lang": config.LANG,
        "dataset": f"{config.DATASET_REPO}:{config.DATASET_FILE}",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (config.INDEX_DIR / f"{index_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nIndex bundle ready in {config.INDEX_DIR}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=None, help="fixed|semantic|recursive|metadata")
    args = ap.parse_args()
    build(args.strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
