"""Extractive generation - the default, budget-fitting answer path.

Picks the sentences from the retrieved chunks that best answer the query and
returns them verbatim. Two properties make this the right default here:

  * Latency. It is pure Python string work, ~1 ms, so the whole pipeline fits
    inside the 200 ms budget. An LLM call cannot: even a fast hosted model is
    several hundred ms of network before it emits a token.
  * Grounding by construction. The answer is a literal span of a retrieved
    passage, so it cannot hallucinate. The grounding guardrail still verifies
    it, but for this path the check is a tripwire rather than a filter.

The tradeoff is fluency: it returns the corpus's own words rather than a
synthesized reply. generation/llm_client.py is the fluent alternative when
latency is not the binding constraint.
"""
from typing import List

import config
from harness.schemas import GenerationResult, RetrievedChunk
from text_utils import keyword_coverage, split_sentences

MAX_ANSWER_CHARS = 400


def _score_sentence(query: str, sentence: str, position: int, chunk_rank: int) -> float:
    """Rank a candidate sentence for how well it answers the query.

    Query-term coverage dominates. Two small priors break ties: earlier
    sentences in a passage tend to carry the definition or headline fact, and
    sentences from better-ranked chunks are more likely to be on topic.
    """
    coverage = keyword_coverage(query, sentence)
    position_prior = 0.05 / (1 + position)
    rank_prior = 0.10 / (1 + chunk_rank)
    # Very short fragments are usually list bullets or headers, not answers.
    length_penalty = 0.25 if len(sentence) < 25 else 0.0
    return coverage + position_prior + rank_prior - length_penalty


def generate(query: str, chunks: List[RetrievedChunk]) -> GenerationResult:
    if not chunks:
        return GenerationResult(
            answer=config.REFUSAL_TEXT, grounded=False, citations=[], mode="extractive"
        )

    scored = []
    for chunk_rank, chunk in enumerate(chunks):
        for position, sentence in enumerate(split_sentences(chunk.text)):
            scored.append(
                (
                    _score_sentence(query, sentence, position, chunk_rank),
                    chunk_rank,
                    position,
                    sentence,
                    chunk,
                )
            )

    if not scored:
        return GenerationResult(
            answer=config.REFUSAL_TEXT, grounded=False, citations=[], mode="extractive"
        )

    scored.sort(key=lambda x: -x[0])
    best_score, best_rank, best_pos, best_sentence, best_chunk = scored[0]

    parts = [best_sentence]
    citations = [best_chunk.chunk_id]

    # A one-line answer often reads as truncated. If the winner is short, append
    # the sentence that physically follows it in the same passage -- adjacent
    # context, not a second-best guess from somewhere else.
    if len(best_sentence) < 120:
        siblings = split_sentences(best_chunk.text)
        if best_pos + 1 < len(siblings):
            parts.append(siblings[best_pos + 1])

    answer = " ".join(parts).strip()
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + " …"

    return GenerationResult(
        answer=answer,
        # Verbatim span of a retrieved chunk, so grounded unless we found
        # nothing that overlapped the query at all.
        grounded=best_score > 0.0,
        citations=citations,
        mode="extractive",
    )
