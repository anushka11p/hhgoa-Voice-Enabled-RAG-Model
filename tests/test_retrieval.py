"""Hybrid retrieval against the real index."""
import config
from tests.conftest import needs_index


@needs_index
def test_recall_meets_the_measured_baseline(engine, corpus):
    """Retrieval quality, asserted in aggregate rather than per-example.

    Any single query can legitimately miss -- measured Recall@1 is ~0.69, so
    asserting a hit on one arbitrary passage would be a coin flip dressed up
    as a test. What must not regress is the aggregate: analytics/run_benchmark
    reports Recall@5 ~0.74, so a floor of 0.55 over 40 queries catches a real
    breakage (a broken index, a mismatched embedding model) without failing on
    noise.
    """
    docs = [d for d in corpus if d.get("source_query")][:40]
    hits = 0
    for doc in docs:
        found = {
            c.metadata.get("passage_id")
            for c in engine.search(doc["source_query"]).chunks
        }
        if doc["passage_id"] in found:
            hits += 1
    recall = hits / len(docs)
    assert recall >= 0.55, f"Recall@k fell to {recall:.2f} over {len(docs)} queries"


@needs_index
def test_returns_at_most_final_top_k(engine):
    result = engine.search("मधुमेह क्या है")
    assert 0 < len(result.chunks) <= config.FINAL_TOP_K


@needs_index
def test_diagnostics_report_every_substage(engine):
    d = engine.search("मधुमेह क्या है").diagnostics
    for key in ("dense_ms", "bm25_ms", "rrf_ms", "rerank_ms", "n_dense", "n_fused"):
        assert key in d, f"missing diagnostic {key}"


@needs_index
def test_hybrid_uses_both_retrievers(engine):
    d = engine.search("मधुमेह क्या है").diagnostics
    assert d["n_dense"] > 0, "dense retrieval returned nothing"
    assert d["n_sparse"] > 0, "BM25 returned nothing"


@needs_index
def test_deadline_pressure_skips_the_reranker(engine):
    # Claim the whole budget is already spent; the engine should serve the
    # fused order rather than blow through it.
    result = engine.search("मधुमेह क्या है", elapsed_ms=config.CORE_BUDGET_MS + 50)
    assert not result.reranked
    assert result.chunks, "skipping rerank must still return results"


@needs_index
def test_normal_query_does_rerank(engine):
    # Pin the cost estimate so this asserts the decision logic, not whatever
    # else happens to be running on the machine.
    engine._rerank_ms_per_candidate = 5.0
    result = engine.search("मधुमेह क्या है", elapsed_ms=0.0)
    assert result.reranked
    assert result.diagnostics["n_reranked"] >= 3


@needs_index
def test_rerank_width_shrinks_as_budget_tightens(engine):
    """Under deadline pressure the reranker narrows before it gives up."""
    engine._rerank_ms_per_candidate = 5.0
    roomy = engine.search("मधुमेह क्या है", elapsed_ms=0.0)
    engine._rerank_ms_per_candidate = 5.0
    tight = engine.search("मधुमेह क्या है", elapsed_ms=140.0)
    assert tight.diagnostics["n_reranked"] < roomy.diagnostics["n_reranked"]


@needs_index
def test_cost_estimate_stays_clamped(engine):
    import config as cfg

    engine._rerank_ms_per_candidate = 1e6  # simulate a pathological spike
    engine.search("मधुमेह क्या है", elapsed_ms=0.0)
    engine._rerank_ms_per_candidate = 5.0
    for _ in range(3):
        engine.search("मधुमेह क्या है", elapsed_ms=0.0)
    assert (
        cfg.RERANK_MS_PER_CANDIDATE_MIN
        <= engine._rerank_ms_per_candidate
        <= cfg.RERANK_MS_PER_CANDIDATE_MAX
    )


@needs_index
def test_topic_similarity_separates_on_and_off_topic(engine, corpus):
    from embeddings.encoder import encode_query

    on = engine.topic_similarity(encode_query(corpus[0]["source_query"]))
    off = engine.topic_similarity(encode_query("book me a flight to Tokyo tomorrow"))
    assert on > off
