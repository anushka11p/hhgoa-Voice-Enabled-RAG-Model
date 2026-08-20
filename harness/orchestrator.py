import time
from harness.schemas import PipelineResult
from stt.sarvam_client import transcribe_with_retry

class PipelineError(Exception):
    pass

def run_pipeline(audio_path: str, retrieve_fn, generate_fn, guardrail_fn, language="hi-IN") -> PipelineResult:
    timings = {}
    t0 = time.perf_counter()

    stage_start = time.perf_counter()
    try:
        stt_result = transcribe_with_retry(audio_path, language)
    except Exception as e:
        raise PipelineError(f"STT stage failed: {e}")
    timings["stt_ms"] = (time.perf_counter() - stage_start) * 1000

    stage_start = time.perf_counter()
    try:
        retrieval_result = retrieve_fn(stt_result.transcript)
    except Exception as e:
        raise PipelineError(f"Retrieval stage failed: {e}")
    timings["retrieval_ms"] = (time.perf_counter() - stage_start) * 1000

    stage_start = time.perf_counter()
    try:
        gen_result = generate_fn(stt_result.transcript, retrieval_result.chunks)
    except Exception as e:
        raise PipelineError(f"Generation stage failed: {e}")
    timings["generation_ms"] = (time.perf_counter() - stage_start) * 1000

    stage_start = time.perf_counter()
    try:
        guard_result = guardrail_fn(gen_result)
    except Exception as e:
        raise PipelineError(f"Guardrail stage failed: {e}")
    timings["guardrail_ms"] = (time.perf_counter() - stage_start) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        transcript=stt_result.transcript,
        answer=guard_result.final_answer,
        allowed=guard_result.allowed,
        stage_timings_ms=timings,
        total_ms=total_ms,
    )