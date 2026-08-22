"""FastAPI backend for the voice RAG demo.

Endpoints:
  GET  /              the demo UI
  POST /api/query     audio in (multipart) -> transcript + grounded answer
  POST /api/ask       text in (JSON)       -> grounded answer, no mic needed
  GET  /api/health    config + index manifest + readiness
  GET  /api/metrics   rolling latency percentiles from the run log

Models and the index load once during startup, not per request: a cold first
query costs seconds of lazy init, and that would land on a real user.
"""
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from harness.logging_utils import log_run, read_percentiles
from harness.orchestrator import PipelineError, run_from_audio, run_from_text
from harness.schemas import QueryRequest
from stt.audio_check import is_likely_silence

STATE = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from retrieval.pipeline import get_engine, warmup

        engine = get_engine()
        warmup(engine)
        STATE["ready"] = True
        STATE["manifest"] = engine.manifest
        print("[server] pipeline warm and ready")
    except Exception as exc:  # noqa: BLE001
        # Boot without an index rather than crash-looping: /api/health then
        # reports exactly what is wrong instead of the container just dying.
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[server] NOT READY -- {STATE['error']}")
    yield


app = FastAPI(title="Voice-Enabled RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _serialize(result, include_chunks: bool = True) -> dict:
    payload = {
        "transcript": result.transcript,
        "answer": result.answer,
        "allowed": result.allowed,
        "grounded": result.grounded,
        "refusal_reason": result.refusal_reason,
        "guardrail_stage": result.guardrail_stage,
        "generation_mode": result.generation_mode,
        "reranked": result.reranked,
        "citations": result.citations,
        "stage_timings_ms": result.stage_timings_ms,
        "core_ms": result.core_ms,
        "total_ms": result.total_ms,
        "within_budget": result.within_budget,
        "budget_ms": config.CORE_BUDGET_MS,
        "diagnostics": result.diagnostics,
    }
    if include_chunks:
        payload["chunks"] = [
            {
                "text": c.text,
                "score": round(c.score, 4),
                "chunk_id": c.chunk_id,
                "passage_id": c.metadata.get("passage_id"),
            }
            for c in result.chunks
        ]
    return payload


def _not_ready():
    return JSONResponse(
        status_code=503,
        content={
            "error": "Pipeline is not ready.",
            "detail": STATE["error"],
            "hint": "Run: python -m data.build_corpus && python -m data.build_index",
        },
    )


@app.post("/api/query")
async def query(audio: UploadFile = File(...)):
    """Voice path: audio file -> transcript -> grounded answer."""
    if not STATE["ready"]:
        return _not_ready()

    suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await audio.read())
            tmp_path = tmp.name

        # Cheapest possible guardrail: reject near-silent audio before paying
        # for an STT round trip. Empty audio otherwise tends to come back as a
        # confidently hallucinated transcript.
        if is_likely_silence(tmp_path):
            return {
                "error": "No speech detected in the audio. Please try again.",
                "guardrail_stage": "silence",
            }

        result = run_from_audio(tmp_path)
        log_run(result, source="voice")
        return _serialize(result)
    except PipelineError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/ask")
async def ask(req: QueryRequest):
    """Text path: same RAG core, no microphone or STT key required."""
    if not STATE["ready"]:
        return _not_ready()
    try:
        result = run_from_text(req.query, mode=req.mode)
        log_run(result, source="text")
        return _serialize(result)
    except PipelineError as exc:
        return JSONResponse(status_code=502, content={"error": str(exc)})


@app.get("/api/health")
def health():
    return {
        "ready": STATE["ready"],
        "error": STATE["error"],
        "config": config.summary(),
        "index": STATE.get("manifest", {}),
    }


@app.get("/api/metrics")
def metrics():
    return read_percentiles()


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
