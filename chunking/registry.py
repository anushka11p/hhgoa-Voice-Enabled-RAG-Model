"""One place to reach every chunking strategy, with a stable output schema.

The four strategy modules (fixed / semantic / recursive / metadata) each own
their own splitting logic. This registry wraps them so callers -- the index
builder and the chunking benchmark -- can swap strategies by name and get back
identically-shaped chunk dicts, with metadata carried through intact.

Why more than one strategy: fixed-size is the cheap baseline, recursive
respects natural language boundaries, semantic keeps whole sentences together
so the embedding sees whole concepts, and metadata-aware bakes the source and
language into the embedded text so filtering signals survive into the vector.
analytics/compare_chunking.py scores all four against the dataset's own
query -> passage labels and that is how ACTIVE_CHUNK_STRATEGY gets chosen.
"""
from typing import Any, Callable, Dict, List

import config
from chunking.fixed import get_fixed_chunks
from chunking.metadata import get_metadata_chunks
from chunking.recursive import get_recursive_chunks
from chunking.semantic import get_semantic_chunks

# Keys the vector store and retrieval layer rely on downstream.
_CARRY_FIELDS = (
    "passage_id",
    "query_id",
    "language",
    "source",
    "source_query",
    "is_selected",
)

STRATEGIES: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "fixed": get_fixed_chunks,
    "semantic": get_semantic_chunks,
    "recursive": get_recursive_chunks,
    "metadata": get_metadata_chunks,
}


def normalize(chunks: List[Dict[str, Any]], strategy: str) -> List[Dict[str, Any]]:
    """Force every strategy's output into one schema.

    The underlying modules copy the whole source document into each chunk, so
    they already carry metadata -- but they vary in which keys they set and
    they can emit empty chunks. This trims to the fields retrieval needs,
    drops blanks, and re-mints globally unique chunk ids (a per-passage
    counter collides once you concatenate passages).
    """
    out: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        text = (ch.get("chunk_text") or "").strip()
        if not text:
            continue
        rec = {k: ch.get(k) for k in _CARRY_FIELDS if k in ch}
        rec["chunk_text"] = text
        rec["chunk_id"] = f"{strategy}:{ch.get('passage_id', 'unknown')}:{i}"
        rec["strategy"] = strategy
        out.append(rec)
    return out


def chunk_documents(
    documents: List[Dict[str, Any]],
    strategy: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Dict[str, Any]]:
    """Run one named strategy over the corpus and return normalized chunks."""
    strategy = (strategy or config.ACTIVE_CHUNK_STRATEGY).lower()
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}. "
            f"Available: {sorted(STRATEGIES)}"
        )
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    raw = STRATEGIES[strategy](
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return normalize(raw, strategy)


def chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """Descriptive stats used in the chunking comparison table."""
    if not chunks:
        return {"n_chunks": 0, "avg_chars": 0.0, "min_chars": 0, "max_chars": 0}
    lengths = [len(c["chunk_text"]) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
