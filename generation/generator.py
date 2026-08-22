"""Routes a query to the configured generation path, with fallback."""
from typing import List, Optional

import config
from generation import extractive
from harness.schemas import GenerationResult, RetrievedChunk


def resolve_mode(requested: Optional[str] = None) -> str:
    mode = (requested or config.GENERATION_MODE).lower()
    if mode == "auto":
        return "llm" if config.ANTHROPIC_API_KEY else "extractive"
    return mode


def generate(
    query: str, chunks: List[RetrievedChunk], mode: Optional[str] = None
) -> GenerationResult:
    """Generate an answer. Falls back to extractive if the LLM path fails.

    The fallback is the point: a generation outage degrades answer fluency,
    it does not take the pipeline down. The returned `mode` records which
    path actually ran, so a silent downgrade is still visible in the logs.
    """
    resolved = resolve_mode(mode)

    if resolved == "llm":
        try:
            from generation import llm_client

            return llm_client.generate(query, chunks)
        except Exception as exc:  # noqa: BLE001 - any LLM failure degrades, never 500s
            result = extractive.generate(query, chunks)
            result.mode = f"extractive (llm fallback: {type(exc).__name__})"
            return result

    return extractive.generate(query, chunks)
