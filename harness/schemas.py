# harness/schemas.py
"""Shared Pydantic contracts between every pipeline stage.

This is the interface contract the three workstreams code against: STT,
retrieval, generation, and guardrails all speak these shapes, so any stage can
be swapped or stubbed without the others noticing.

Fields added after the day-1 contract are optional with defaults, so older
call sites (and harness/stubs.py) keep working unchanged.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class STTResult(BaseModel):
    transcript: str
    language: str
    confidence: float
    duration_ms: float


class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict
    chunk_id: str = ""


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    retrieval_ms: float
    # Set False when the orchestrator dropped the cross-encoder to stay inside
    # the latency budget -- surfaced so a slow run is visible, not silent.
    reranked: bool = True
    # Score telemetry the guardrails threshold against (top cosine, top
    # cross-encoder score, nearest-centroid similarity, per-stage timings).
    diagnostics: dict = Field(default_factory=dict)


class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    citations: list[str]
    mode: str = "extractive"


class GuardrailResult(BaseModel):
    allowed: bool
    reason: Optional[str]
    final_answer: str
    # Which gate fired: unsafe | off_topic | low_confidence | ungrounded | None
    stage: Optional[str] = None
    checks: dict = Field(default_factory=dict)


class PipelineResult(BaseModel):
    transcript: str
    answer: str
    allowed: bool
    stage_timings_ms: dict
    total_ms: float
    # --- added after the day-1 contract, all optional ---
    language: str = ""
    citations: list[str] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    grounded: bool = True
    refusal_reason: Optional[str] = None
    guardrail_stage: Optional[str] = None
    generation_mode: str = "extractive"
    reranked: bool = True
    # Core = retrieval + generation + guardrails, i.e. everything the 200 ms
    # budget covers. Excludes the STT network round trip.
    core_ms: float = 0.0
    within_budget: bool = True
    diagnostics: dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """Text-mode entry point, used by the benchmark and the /api/ask endpoint."""
    query: str
    top_k: int = 4
    mode: Optional[str] = None
    filters: dict[str, Any] = Field(default_factory=dict)
