"""Claude-backed generation - the fluent, higher-latency answer path.

Used when GENERATION_MODE is "llm" (or "auto" with a key present). It trades
the 200 ms budget for a synthesized answer instead of a verbatim span, so the
benchmark reports it separately from the extractive default.

Grounding is defended three ways: a context-only system prompt, a parsed
citation back to the passage used, and -- because prompts are not a security
boundary -- the independent overlap check in guardrails/grounding.py that runs
on the output regardless of what the model claims.
"""
import re
from typing import List, Optional

import config
from generation.prompt import build_messages
from harness.schemas import GenerationResult, RetrievedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")
_client = None


class LLMUnavailable(RuntimeError):
    pass


def get_client():
    """Build the Anthropic client once.

    Retries and timeouts are the SDK's own: it already retries connection
    errors, 408/409/429 and 5xx with exponential backoff, so wrapping it in
    another retry loop would just multiply the wall-clock ceiling.
    """
    global _client
    if _client is None:
        if not config.ANTHROPIC_API_KEY:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set. Use GENERATION_MODE=extractive "
                "or add the key to .env."
            )
        import anthropic

        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            timeout=config.LLM_TIMEOUT_S,
            max_retries=config.STT_MAX_RETRIES,
        )
    return _client


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def _resolve_citations(text: str, chunks: List[RetrievedChunk]) -> List[str]:
    """Map the [n] markers the model emitted back to real chunk ids."""
    ids: List[str] = []
    for marker in _CITATION_RE.findall(text):
        idx = int(marker) - 1
        if 0 <= idx < len(chunks):
            cid = chunks[idx].chunk_id
            if cid not in ids:
                ids.append(cid)
    return ids


def generate(
    query: str, chunks: List[RetrievedChunk], model: Optional[str] = None
) -> GenerationResult:
    if not chunks:
        return GenerationResult(
            answer=config.REFUSAL_TEXT, grounded=False, citations=[], mode="llm"
        )

    client = get_client()
    system, user = build_messages(query, chunks, config.LANG, config.REFUSAL_TEXT)

    response = client.messages.create(
        model=model or config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        # This is short, closed-book extraction from 4 short passages -- the
        # cheapest effort setting is the right one, and it keeps the latency
        # gap versus the extractive path as small as it can be.
        output_config={"effort": "low"},
    )

    # A safety refusal from the model is not an error here: this pipeline
    # already has a refusal path, and "decline to answer" is the correct
    # product behaviour. So we surface our own refusal text rather than
    # routing to a fallback model.
    if response.stop_reason == "refusal":
        return GenerationResult(
            answer=config.REFUSAL_TEXT, grounded=False, citations=[], mode="llm"
        )

    raw = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if not raw:
        return GenerationResult(
            answer=config.REFUSAL_TEXT, grounded=False, citations=[], mode="llm"
        )

    citations = _resolve_citations(raw, chunks)
    answer = _strip_citations(raw)
    refused = answer.strip() == config.REFUSAL_TEXT.strip()

    return GenerationResult(
        answer=answer or config.REFUSAL_TEXT,
        # Provisional only. guardrails/grounding.py re-checks this against the
        # passages and can still overturn it.
        grounded=not refused,
        citations=citations or [c.chunk_id for c in chunks[:1]],
        mode="llm",
    )
