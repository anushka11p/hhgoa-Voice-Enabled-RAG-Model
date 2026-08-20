from harness.schemas import RetrievalResult, RetrievedChunk, GenerationResult, GuardrailResult

def stub_retrieve(query: str) -> RetrievalResult:
    return RetrievalResult(
        chunks=[RetrievedChunk(text="stub passage", score=0.9, metadata={})],
        retrieval_ms=5.0,
    )

def stub_generate(query: str, chunks: list) -> GenerationResult:
    return GenerationResult(answer=f"Stub answer for: {query}", grounded=True, citations=[])

def stub_guardrail(gen_result) -> GuardrailResult:
    return GuardrailResult(allowed=True, reason=None, final_answer=gen_result.answer)