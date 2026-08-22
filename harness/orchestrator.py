"""The harness: stage sequencing, timing, guardrail placement, error recovery.

Pipeline shape:

    audio -> [silence gate] -> STT -> [pre guardrails] -> retrieval
          -> generation -> [post guardrails] -> answer

Three things this module owns that the individual stages deliberately do not:

  * Budget accounting. Every stage is timed. `core_ms` sums retrieval +
    generation + guardrails -- the part the 200 ms target actually covers --
    and `within_budget` records whether we made it. STT is timed too but kept
    out of `core_ms`, because it is a network round trip to a third-party API
    and no amount of local engineering brings it under 200 ms.

  * Short-circuiting. A guardrail refusal before retrieval returns immediately
    rather than running work whose output would be discarded.

  * Degradation over failure. Retrieval passes its elapsed time down so the
    reranker can be dropped under deadline pressure; a generation failure
    falls back to the extractive path. The pipeline returns a worse answer
    rather than no answer, and always records that it did.
"""
import time
from typing import Optional

import config
from harness.schemas import PipelineResult


class PipelineError(Exception):
    """Raised only when a stage fails unrecoverably."""


class _Timer:
    """Accumulates per-stage timings in milliseconds."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.stages: dict = {}

    def mark(self, name: str, start: float) -> float:
        ms = (time.perf_counter() - start) * 1000
        self.stages[name] = round(ms, 3)
        return ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    def core_ms(self) -> float:
        """Everything the 200 ms budget covers -- i.e. all but STT."""
        return round(
            sum(v for k, v in self.stages.items() if not k.startswith("stt")), 3
        )


def _finish(
    timer: _Timer,
    transcript: str,
    guard,
    gen=None,
    retrieval=None,
    language: str = "",
) -> PipelineResult:
    core = timer.core_ms()
    diagnostics = dict(retrieval.diagnostics) if retrieval else {}
    diagnostics["guardrail_checks"] = guard.checks
    return PipelineResult(
        transcript=transcript,
        answer=guard.final_answer,
        allowed=guard.allowed,
        stage_timings_ms=timer.stages,
        total_ms=round(timer.total_ms, 3),
        language=language,
        citations=(gen.citations if gen and guard.allowed else []),
        chunks=(retrieval.chunks if retrieval and guard.allowed else []),
        grounded=(gen.grounded if gen else False),
        refusal_reason=guard.reason,
        guardrail_stage=guard.stage,
        generation_mode=(gen.mode if gen else "none"),
        reranked=(retrieval.reranked if retrieval else False),
        core_ms=core,
        within_budget=core <= config.CORE_BUDGET_MS,
        diagnostics=diagnostics,
    )


def run_from_text(query: str, mode: Optional[str] = None, language: str = "") -> PipelineResult:
    """Run the RAG core on an already-transcribed query.

    This is the path the latency benchmark measures and the /api/ask endpoint
    serves, because it isolates the part of the system we actually control.
    """
    from embeddings.encoder import encode_query
    from generation.generator import generate
    from guardrails import pipeline as guards
    from retrieval.pipeline import get_engine

    timer = _Timer()
    engine = get_engine()

    # Embed once, up front: the off-topic gate and the dense search both need
    # the same vector, and encoding is ~10 ms we should not pay twice.
    t = time.perf_counter()
    try:
        qvec = encode_query(query) if query and query.strip() else None
        centroid_sim = engine.topic_similarity(qvec) if qvec is not None else None
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Query embedding failed: {exc}") from exc
    embed_ms = timer.mark("embed_ms", t)

    # --- guardrails, before any expensive work ---
    t = time.perf_counter()
    blocked = guards.pre_check(query, centroid_sim)
    timer.mark("guardrail_pre_ms", t)
    if blocked is not None:
        return _finish(timer, query, blocked, language=language)

    # --- retrieval ---
    t = time.perf_counter()
    try:
        retrieval = engine.search(
            query,
            qvec=qvec,
            # Hand the reranker our elapsed budget so it can bow out if we are
            # already running late.
            elapsed_ms=embed_ms + timer.stages.get("guardrail_pre_ms", 0.0),
        )
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Retrieval stage failed: {exc}") from exc
    timer.mark("retrieval_ms", t)
    # The engine timed its own embed of the query; we already paid it above.
    retrieval.diagnostics.pop("embed_ms", None)

    # --- generation ---
    t = time.perf_counter()
    try:
        gen = generate(query, retrieval.chunks, mode=mode)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Generation stage failed: {exc}") from exc
    timer.mark("generation_ms", t)

    # --- guardrails, after generation ---
    t = time.perf_counter()
    try:
        guard = guards.post_check(query, retrieval, gen)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Guardrail stage failed: {exc}") from exc
    timer.mark("guardrail_post_ms", t)

    return _finish(timer, query, guard, gen=gen, retrieval=retrieval, language=language)


def run_from_audio(
    audio_path: str, language: Optional[str] = None, mode: Optional[str] = None
) -> PipelineResult:
    """Full voice path: transcribe, then run the RAG core."""
    from stt.sarvam_client import transcribe_with_retry

    language = language or config.STT_LANGUAGE_CODE

    stt_start = time.perf_counter()
    try:
        stt = transcribe_with_retry(audio_path, language)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"STT stage failed: {exc}") from exc
    stt_ms = round((time.perf_counter() - stt_start) * 1000, 3)

    result = run_from_text(stt.transcript, mode=mode, language=stt.language)

    # Splice STT in front of the core timings without letting it count toward
    # the core budget.
    result.stage_timings_ms = {"stt_ms": stt_ms, **result.stage_timings_ms}
    result.total_ms = round(result.total_ms + stt_ms, 3)
    return result


def run_pipeline(
    audio_path: str,
    retrieve_fn=None,
    generate_fn=None,
    guardrail_fn=None,
    language: str = None,
) -> PipelineResult:
    """Backwards-compatible entry point.

    The day-1 harness took injected retrieve/generate/guardrail callables so
    the pipeline could run end to end against stubs before the real modules
    existed. That contract still works -- harness/stubs.py and the original
    tests depend on it. Called without callables, it runs the real pipeline.
    """
    if retrieve_fn is None and generate_fn is None and guardrail_fn is None:
        return run_from_audio(audio_path, language=language)

    from stt.sarvam_client import transcribe_with_retry

    timer = _Timer()
    t = time.perf_counter()
    try:
        stt = transcribe_with_retry(audio_path, language or config.STT_LANGUAGE_CODE)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"STT stage failed: {exc}") from exc
    timer.mark("stt_ms", t)

    t = time.perf_counter()
    try:
        retrieval = retrieve_fn(stt.transcript)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Retrieval stage failed: {exc}") from exc
    timer.mark("retrieval_ms", t)

    t = time.perf_counter()
    try:
        gen = generate_fn(stt.transcript, retrieval.chunks)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Generation stage failed: {exc}") from exc
    timer.mark("generation_ms", t)

    t = time.perf_counter()
    try:
        guard = guardrail_fn(gen)
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Guardrail stage failed: {exc}") from exc
    timer.mark("guardrail_ms", t)

    return _finish(
        timer, stt.transcript, guard, gen=gen, retrieval=retrieval, language=stt.language
    )
