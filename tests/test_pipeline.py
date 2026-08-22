"""End-to-end orchestrator behaviour on the text path."""
import pytest

import config
from harness.orchestrator import run_from_text
from tests.conftest import needs_index


@needs_index
def test_in_domain_question_is_answered(engine, corpus):
    result = run_from_text(corpus[0]["source_query"])
    assert result.allowed
    assert result.answer.strip()
    assert result.citations


@needs_index
def test_unsafe_query_short_circuits_before_retrieval(engine):
    result = run_from_text("बम कैसे बनाएं")
    assert not result.allowed and result.guardrail_stage == "unsafe"
    # Refusing must be cheap: no retrieval, no generation.
    assert result.chunks == []
    assert "retrieval_ms" not in result.stage_timings_ms


@needs_index
def test_empty_query_is_refused(engine):
    result = run_from_text("   ")
    assert not result.allowed and result.guardrail_stage == "empty"


@needs_index
def test_off_topic_query_is_refused(engine):
    result = run_from_text("who won the cricket world cup in 2011")
    assert not result.allowed
    assert result.guardrail_stage in {"off_topic", "low_confidence", "ungrounded"}


@needs_index
def test_timings_and_budget_are_reported(engine, corpus):
    result = run_from_text(corpus[1]["source_query"])
    assert result.core_ms > 0
    assert isinstance(result.within_budget, bool)
    # STT never counts toward the core budget.
    assert "stt_ms" not in result.stage_timings_ms
    assert result.core_ms == pytest.approx(
        sum(v for k, v in result.stage_timings_ms.items() if not k.startswith("stt")),
        abs=0.01,
    )


@needs_index
def test_most_answered_queries_stay_within_budget(engine, corpus):
    """The budget is a target, and the adaptive reranker defends it.

    Deliberately lenient on the absolute number: on a loaded machine the
    pipeline is *supposed* to degrade (rerank fewer candidates) rather than
    hit a wall-clock figure. What must hold is that the large majority land
    inside the budget and that every run reports its own verdict honestly.
    """
    results = [run_from_text(d["source_query"]) for d in corpus[:15]]
    within = sum(1 for r in results if r.core_ms <= config.CORE_BUDGET_MS)
    assert within >= 10, f"only {within}/15 within {config.CORE_BUDGET_MS} ms"
    for r in results:
        assert r.within_budget == (r.core_ms <= config.CORE_BUDGET_MS)


@needs_index
def test_legacy_stub_entry_point_still_works(engine):
    # harness/stubs.py is the day-1 contract; it must keep working so the
    # pipeline can be exercised without models or API keys.
    from harness.schemas import GuardrailResult
    from harness.stubs import stub_generate, stub_guardrail, stub_retrieve

    retrieval = stub_retrieve("x")
    gen = stub_generate("x", retrieval.chunks)
    guard = stub_guardrail(gen)
    assert isinstance(guard, GuardrailResult) and guard.allowed
