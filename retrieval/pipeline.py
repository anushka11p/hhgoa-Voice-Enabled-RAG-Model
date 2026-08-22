"""Serving-time hybrid retrieval: dense + BM25 -> RRF -> cross-encoder rerank.

Loads the prebuilt index bundle once at boot and answers queries against it.
Everything here is on the latency-critical path, so:

  * the index is memory-resident (no disk or network I/O per query),
  * every sub-stage is timed individually and reported in diagnostics,
  * the cross-encoder -- by far the most expensive stage -- can be skipped
    under deadline pressure, falling back to the RRF-fused order.

Why hybrid rather than dense alone: dense embeddings match meaning but lose
exact tokens (numbers, names, units, transliterated loanwords), which matters
a lot in Hindi MS MARCO where the answer often hinges on a specific figure.
BM25 catches those. RRF merges the two rankings without needing their score
scales to be comparable, and the cross-encoder then rescores the shortlist by
reading query and passage together instead of comparing two fixed vectors.
"""
import json
import pickle
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

import config
from embeddings.encoder import encode_query, l2_to_cosine
from harness.schemas import RetrievalResult, RetrievedChunk
from retrieval.rrf import compute_rrf


class IndexNotBuilt(RuntimeError):
    pass


class RetrievalEngine:
    """Holds the FAISS index, BM25 model, topic centroids, and reranker."""

    def __init__(self, load_reranker: bool = True, index_name: str = None):
        import faiss

        # index_name lets the chunking benchmark load alternate bundles
        # without disturbing the one the server serves.
        name = index_name or config.INDEX_NAME
        base = config.INDEX_DIR
        idx_path = base / f"{name}.bin"
        meta_path = base / f"{name}_meta.json"
        bm25_path = base / f"{name}_bm25.pkl"
        topic_path = base / f"{name}_topic.npz"

        if not idx_path.exists():
            raise IndexNotBuilt(
                f"No index at {idx_path}. Run:\n"
                "  python -m data.build_corpus && python -m data.build_index"
            )

        print(f"[retrieval] loading index from {base} ...")
        self.index = faiss.read_index(str(idx_path))
        self.metadata: List[Dict[str, Any]] = json.loads(
            meta_path.read_text(encoding="utf-8")
        )
        with open(bm25_path, "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["model"]
        self.centroids = np.load(topic_path)["centroids"].astype(np.float32)

        manifest_path = base / f"{name}_manifest.json"
        self.manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )

        self.reranker = None
        if load_reranker:
            from sentence_transformers import CrossEncoder

            print(f"[retrieval] loading reranker {config.RERANK_MODEL} ...")
            # Cap sequence length: chunks are ~400 chars, so 256 tokens covers
            # them, and it bounds the worst case when a long chunk slips
            # through instead of letting one query blow the budget.
            self.reranker = CrossEncoder(config.RERANK_MODEL, max_length=256)

        # Measured cost of reranking ONE candidate, in ms. Seeded from a
        # rough prior and then tracked as an EWMA of real observations, so the
        # pipeline calibrates itself to whatever hardware it lands on rather
        # than trusting a number tuned on a developer laptop.
        self._rerank_ms_per_candidate = config.RERANK_MS_PER_CANDIDATE_SEED

        self._lock = threading.Lock()
        print(
            f"[retrieval] ready: {self.index.ntotal} chunks, "
            f"strategy={self.manifest.get('strategy')}"
        )

    # ------------------------------------------------------------------
    # Guardrail support
    # ------------------------------------------------------------------
    def topic_similarity(self, qvec: np.ndarray) -> float:
        """Cosine to the nearest corpus centroid.

        A cheap pre-retrieval "is this even about our corpus?" signal: one
        (24, 384) matmul, microseconds, no index scan.
        """
        return float(np.max(self.centroids @ qvec))

    # ------------------------------------------------------------------
    # Retrieval stages
    # ------------------------------------------------------------------
    def _dense(self, qvec: np.ndarray, top_k: int):
        distances, indices = self.index.search(
            qvec.reshape(1, -1).astype(np.float32), top_k
        )
        out = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            # Unit vectors -> monotone map from L2 distance to cosine.
            out.append((meta["chunk_id"], l2_to_cosine(float(dist)), meta))
        return out

    def _sparse(self, query: str, top_k: int):
        tokens = query.lower().split()
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        out = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            meta = self.metadata[int(i)]
            out.append((meta["chunk_id"], float(scores[i]), meta))
        return out

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int):
        pairs = [[query, c.get("chunk_text", "")] for c in candidates]
        with self._lock:  # CrossEncoder is not thread-safe
            scores = self.reranker.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["relevance_score"] = float(s)
        return sorted(candidates, key=lambda d: d["relevance_score"], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = None,
        qvec: Optional[np.ndarray] = None,
        allow_rerank: bool = True,
        elapsed_ms: float = 0.0,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval for one query.

        `elapsed_ms` is how much of the core budget the caller has already
        spent; if adding a rerank would blow past RERANK_SKIP_THRESHOLD_MS we
        serve the fused order instead. That degradation is reported, never
        hidden.
        """
        top_k = top_k or config.FINAL_TOP_K
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        t = time.perf_counter()
        if qvec is None:
            qvec = encode_query(query)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense = self._dense(qvec, config.DENSE_TOP_K)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        sparse = self._sparse(query, config.BM25_TOP_K)
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        fused = compute_rrf(dense, sparse, k=config.RRF_K, top_n=config.RRF_TOP_N)
        timings["rrf_ms"] = (time.perf_counter() - t) * 1000

        # --- budget-aware reranking -------------------------------------
        # The cross-encoder is the only stage big enough to break the budget,
        # and its cost is close to linear in the number of candidates. So
        # instead of an all-or-nothing skip, spend exactly the candidates the
        # remaining budget affords. A slow query reranks 4 instead of 8 and
        # still gets reranked; only a query with no headroom left skips
        # entirely.
        spent = elapsed_ms + (time.perf_counter() - t_start) * 1000
        remaining = config.CORE_BUDGET_MS - spent - config.POST_RERANK_RESERVE_MS
        affordable = int(remaining / max(self._rerank_ms_per_candidate, 1e-6))
        n_candidates = max(0, min(len(fused), config.RRF_TOP_N, affordable))

        do_rerank = (
            allow_rerank
            and self.reranker is not None
            and n_candidates >= config.MIN_RERANK_CANDIDATES
        )

        if do_rerank:
            t = time.perf_counter()
            final = self._rerank(query, fused[:n_candidates], top_k)
            rerank_ms = (time.perf_counter() - t) * 1000
            timings["rerank_ms"] = rerank_ms
            # EWMA update, so the estimate tracks the machine we are on.
            # Asymmetric on purpose: rise slowly (one spike should not collapse
            # rerank width) but fall quickly (recover as soon as load clears),
            # then clamp so the estimate can never run away in either direction.
            observed = rerank_ms / n_candidates
            alpha = 0.2 if observed > self._rerank_ms_per_candidate else 0.5
            self._rerank_ms_per_candidate = min(
                config.RERANK_MS_PER_CANDIDATE_MAX,
                max(
                    config.RERANK_MS_PER_CANDIDATE_MIN,
                    (1 - alpha) * self._rerank_ms_per_candidate + alpha * observed,
                ),
            )
        else:
            final = fused[:top_k]
            timings["rerank_ms"] = 0.0
            n_candidates = 0

        dense_cos = {cid: sc for cid, sc, _ in dense}
        chunks = [
            RetrievedChunk(
                text=d.get("chunk_text", ""),
                # Cross-encoder score when we reranked, RRF score when we did
                # not -- `reranked` tells the caller which scale this is on.
                score=float(
                    d.get("relevance_score", d.get("rrf_score", 0.0))
                    if do_rerank
                    else d.get("rrf_score", 0.0)
                ),
                chunk_id=d.get("chunk_id", ""),
                metadata={
                    "passage_id": d.get("passage_id"),
                    "query_id": d.get("query_id"),
                    "language": d.get("language"),
                    "source": d.get("source"),
                    "strategy": d.get("strategy"),
                    "rrf_score": d.get("rrf_score"),
                    "dense_cosine": dense_cos.get(d.get("chunk_id")),
                },
            )
            for d in final
        ]

        total_ms = (time.perf_counter() - t_start) * 1000
        diagnostics = {
            **{k: round(v, 3) for k, v in timings.items()},
            "n_dense": len(dense),
            "n_sparse": len(sparse),
            "n_fused": len(fused),
            "n_reranked": n_candidates,
            "rerank_ms_per_candidate": round(self._rerank_ms_per_candidate, 3),
            "top_dense_cosine": round(max(dense_cos.values()), 4) if dense_cos else 0.0,
            "top_rerank_score": (
                round(chunks[0].score, 4) if (chunks and do_rerank) else None
            ),
        }
        return RetrievalResult(
            chunks=chunks,
            retrieval_ms=total_ms,
            reranked=do_rerank,
            diagnostics=diagnostics,
        )


# Process-wide singleton so the server loads the index exactly once.
_engine: Optional[RetrievalEngine] = None
_engine_lock = threading.Lock()


def get_engine(load_reranker: bool = True) -> RetrievalEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RetrievalEngine(load_reranker=load_reranker)
    return _engine


def warmup(engine: Optional[RetrievalEngine] = None) -> None:
    """Run one throwaway query through every stage.

    Both transformer models lazily allocate buffers and JIT-select kernels on
    their first forward pass. Measured cold, that first query costs seconds;
    warm, it costs tens of milliseconds. Serving and benchmarking both call
    this at boot so the first real user is not the one who pays for it.
    """
    engine = engine or get_engine()
    engine.search("वार्म अप प्रश्न", top_k=2)
