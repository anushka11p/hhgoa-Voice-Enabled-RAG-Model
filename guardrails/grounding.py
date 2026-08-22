"""Grounding check - does the answer actually follow from the passages?

Runs on every generated answer regardless of which generation path produced
it. For the extractive path the answer is a verbatim span, so this is a
tripwire that should always pass; for the LLM path it is a real filter, and
the one that catches hallucination.

Method: take the answer's content tokens (stopwords stripped) and measure what
fraction appear in the passages actually cited. An answer that invents a
number, a name, or a claim drops below the threshold and gets refused.

Why not an NLI/entailment model: a cross-encoder entailment pass is another
40-90 ms on a 200 ms budget, and it would be the second-largest cost in the
pipeline. Lexical overlap is a weaker signal but it is the right weak signal
here -- extractive answers are spans, and the failure mode we care about
(fabricated specifics) is exactly what token overlap catches. The threshold is
tunable via GROUNDING_OVERLAP_THRESHOLD.
"""
from typing import List, Tuple

import config
from harness.schemas import RetrievedChunk
from text_utils import overlap_ratio


def check(
    answer: str, chunks: List[RetrievedChunk], citations: List[str]
) -> Tuple[bool, float, str]:
    """Return (is_grounded, overlap, reason)."""
    if not answer.strip():
        return False, 0.0, "empty answer"
    if not chunks:
        return False, 0.0, "no passages to ground against"

    # Prefer the passages the generator said it used; fall back to everything
    # retrieved if the citations did not resolve.
    cited = [c for c in chunks if c.chunk_id in set(citations)] or list(chunks)
    source = " ".join(c.text for c in cited)

    overlap = overlap_ratio(answer, source)
    if overlap < config.GROUNDING_OVERLAP_THRESHOLD:
        return False, overlap, (
            f"only {overlap:.0%} of the answer's content words appear in the "
            f"cited passages (floor {config.GROUNDING_OVERLAP_THRESHOLD:.0%})"
        )
    return True, overlap, ""
