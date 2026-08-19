# harness/schemas.py
from pydantic import BaseModel
from typing import Optional

class STTResult(BaseModel):
    transcript: str
    language: str
    confidence: float
    duration_ms: float

class RetrievedChunk(BaseModel):
    text: str
    score: float
    metadata: dict

class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    retrieval_ms: float

class GenerationResult(BaseModel):
    answer: str
    grounded: bool
    citations: list[str]

class GuardrailResult(BaseModel):
    allowed: bool
    reason: Optional[str]
    final_answer: str

class PipelineResult(BaseModel):
    transcript: str
    answer: str
    allowed: bool
    stage_timings_ms: dict
    total_ms: float