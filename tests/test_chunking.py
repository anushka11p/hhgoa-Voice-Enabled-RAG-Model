"""Every chunking strategy must produce usable, uniquely-identified chunks."""
import pytest

from chunking.registry import STRATEGIES, chunk_documents, chunk_stats

DOCS = [
    {
        "passage_id": "p1",
        "query_id": "q1",
        "language": "hi",
        "source": "IndicMSMARCO",
        "source_query": "टमाटर क्या है",
        "passage_text": (
            "विरासती बागवानों को यह बात पता है। खुले परागण की अवधारणा थोड़ी "
            "भ्रामक है। उदाहरण के लिए स्क्वैश और कद्दू लें। इस क्षेत्र के "
            "विशेषज्ञ इस बात से सहमत हैं कि यह जटिल है।"
        ) * 3,
    }
]


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_strategy_produces_chunks(strategy):
    chunks = chunk_documents(DOCS, strategy=strategy, chunk_size=200, chunk_overlap=40)
    assert chunks, f"{strategy} produced nothing"
    assert all(c["chunk_text"].strip() for c in chunks)


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_chunk_ids_are_unique(strategy):
    chunks = chunk_documents(DOCS, strategy=strategy, chunk_size=200, chunk_overlap=40)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_metadata_is_carried_through(strategy):
    chunks = chunk_documents(DOCS, strategy=strategy, chunk_size=200, chunk_overlap=40)
    assert all(c["passage_id"] == "p1" for c in chunks)
    assert all(c["strategy"] == strategy for c in chunks)


def test_hindi_actually_splits():
    # A splitter that only knows [.!?] returns one giant chunk on Devanagari.
    chunks = chunk_documents(DOCS, strategy="semantic", chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_documents(DOCS, strategy="does-not-exist")


def test_chunk_stats_on_empty():
    assert chunk_stats([])["n_chunks"] == 0
