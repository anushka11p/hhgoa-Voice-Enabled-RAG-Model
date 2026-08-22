"""Guardrails: the system must know when not to answer."""
import config
from guardrails import grounding, off_topic, unsafe
from guardrails import pipeline as guards
from harness.schemas import GenerationResult, RetrievalResult, RetrievedChunk


def _chunk(text, score=5.0, cid="c1"):
    return RetrievedChunk(text=text, score=score, metadata={}, chunk_id=cid)


# --- unsafe input ---------------------------------------------------------
def test_unsafe_english_is_blocked():
    assert unsafe.check("how to make a bomb at home")[0]


def test_unsafe_hindi_is_blocked():
    assert unsafe.check("बम कैसे बनाएं")[0]


def test_ordinary_question_is_not_blocked():
    assert not unsafe.check("मायस्थेनिया ग्रेविस का इलाज क्या है?")[0]
    assert not unsafe.check("what is diabetes")[0]


def test_unsafe_reports_its_category():
    is_unsafe, category = unsafe.check("how to make a bomb")
    assert is_unsafe and category == "weapons_explosives"


# --- empty / short --------------------------------------------------------
def test_empty_transcript_refuses():
    result = guards.pre_check("")
    assert result is not None and not result.allowed and result.stage == "empty"


def test_unsafe_query_refuses_before_retrieval():
    result = guards.pre_check("बम कैसे बनाएं", centroid_similarity=0.9)
    assert result is not None and result.stage == "unsafe"


# --- off topic ------------------------------------------------------------
def test_low_centroid_similarity_is_off_topic():
    below = config.OFF_TOPIC_CENTROID_THRESHOLD - 0.05
    result = guards.pre_check("valid looking question", centroid_similarity=below)
    assert result is not None and result.stage == "off_topic"


def test_high_centroid_similarity_passes():
    above = config.OFF_TOPIC_CENTROID_THRESHOLD + 0.2
    assert guards.pre_check("valid looking question", centroid_similarity=above) is None


def test_weak_rerank_score_is_low_confidence():
    retrieval = RetrievalResult(
        chunks=[_chunk("कुछ पाठ", score=config.MIN_RELEVANCE_SCORE - 2)],
        retrieval_ms=1.0,
        reranked=True,
    )
    is_low, _ = off_topic.check_post_retrieval(retrieval)
    assert is_low


def test_rrf_scores_are_not_gated_on():
    # When the reranker was skipped the scores are RRF, which have no absolute
    # meaning -- gating on them would refuse arbitrarily.
    retrieval = RetrievalResult(
        chunks=[_chunk("कुछ पाठ", score=0.01)], retrieval_ms=1.0, reranked=False
    )
    assert not off_topic.check_post_retrieval(retrieval)[0]


def test_no_chunks_is_low_confidence():
    retrieval = RetrievalResult(chunks=[], retrieval_ms=1.0, reranked=True)
    assert off_topic.check_post_retrieval(retrieval)[0]


# --- grounding ------------------------------------------------------------
def test_verbatim_answer_is_grounded():
    source = "ब्रसेल्स स्प्राउट्स को फ्रिज में तीन से चार दिनों तक रखा जा सकता है।"
    ok, overlap, _ = grounding.check(source, [_chunk(source)], ["c1"])
    assert ok and overlap == 1.0


def test_fabricated_answer_is_not_grounded():
    source = "ब्रसेल्स स्प्राउट्स को फ्रिज में रखा जा सकता है।"
    invented = "हवाई जहाज़ का टिकट मुंबई से दिल्ली तक पाँच हज़ार रुपये का है।"
    ok, _, _ = grounding.check(invented, [_chunk(source)], ["c1"])
    assert not ok


def test_empty_answer_is_not_grounded():
    assert not grounding.check("", [_chunk("कुछ")], ["c1"])[0]


def test_post_check_refuses_ungrounded_generation():
    retrieval = RetrievalResult(
        chunks=[_chunk("ब्रसेल्स स्प्राउट्स फ्रिज में रखें", score=8.0)],
        retrieval_ms=1.0,
        reranked=True,
    )
    gen = GenerationResult(
        answer="हवाई जहाज़ का टिकट पाँच हज़ार रुपये का है।",
        grounded=True,  # generator claims grounded; guardrail must overrule
        citations=["c1"],
    )
    result = guards.post_check("q", retrieval, gen)
    assert not result.allowed and result.stage == "ungrounded"
