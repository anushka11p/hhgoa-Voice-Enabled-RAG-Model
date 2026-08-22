"""Sequences the guardrails around the pipeline.

Two entry points, deliberately placed on either side of the expensive work:

  pre_check(...)   runs before retrieval. Everything it can reject, it rejects
                   for free -- empty transcripts, unsafe asks, and queries that
                   are nowhere near the corpus. A refusal here costs ~0.2 ms
                   instead of a full retrieve-and-generate.

  post_check(...)  runs after generation. Decides whether the answer we built
                   is good enough to show: is the best passage actually
                   relevant, and is the answer supported by it.

Both return a GuardrailResult, so the caller never has to know which specific
gate fired -- it reads `allowed`, shows `final_answer`, and logs `stage`.
"""
from typing import List, Optional

import config
from guardrails import grounding, off_topic, refusal, unsafe
from harness.schemas import GenerationResult, GuardrailResult, RetrievedChunk


def pre_check(query: str, centroid_similarity: Optional[float] = None) -> Optional[GuardrailResult]:
    """Gates that run before retrieval. Returns None when the query may proceed."""
    checks = {}

    if not query or len(query.strip()) < config.MIN_QUERY_CHARS:
        return GuardrailResult(
            allowed=False,
            reason="transcript was empty or too short to be a question",
            final_answer=refusal.message("empty"),
            stage="empty",
            checks={"query_chars": len(query.strip() if query else "")},
        )

    is_unsafe, category = unsafe.check(query)
    checks["unsafe_category"] = category
    if is_unsafe:
        return GuardrailResult(
            allowed=False,
            reason=f"unsafe input detected (category: {category})",
            final_answer=refusal.message("unsafe"),
            stage="unsafe",
            checks=checks,
        )

    if centroid_similarity is not None:
        checks["centroid_similarity"] = round(centroid_similarity, 4)
        is_off, why = off_topic.check_pre_retrieval(centroid_similarity)
        if is_off:
            return GuardrailResult(
                allowed=False,
                reason=why,
                final_answer=refusal.message("off_topic"),
                stage="off_topic",
                checks=checks,
            )

    return None


def post_check(
    query: str,
    retrieval,
    gen_result: GenerationResult,
    chunks: List[RetrievedChunk] = None,
) -> GuardrailResult:
    """Gates that run after generation."""
    chunks = chunks if chunks is not None else retrieval.chunks
    checks = {"reranked": retrieval.reranked}

    low_conf, why = off_topic.check_post_retrieval(retrieval)
    if low_conf:
        return GuardrailResult(
            allowed=False,
            reason=why,
            final_answer=refusal.message("low_confidence"),
            stage="low_confidence",
            checks=checks,
        )

    is_grounded, overlap, why = grounding.check(
        gen_result.answer, chunks, gen_result.citations
    )
    checks["grounding_overlap"] = round(overlap, 4)
    if not is_grounded:
        return GuardrailResult(
            allowed=False,
            reason=why,
            final_answer=refusal.message("ungrounded"),
            stage="ungrounded",
            checks=checks,
        )

    return GuardrailResult(
        allowed=True,
        reason=None,
        final_answer=gen_result.answer,
        stage=None,
        checks=checks,
    )
