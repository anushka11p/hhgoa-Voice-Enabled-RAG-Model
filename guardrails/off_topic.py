"""Off-topic detection, in two stages either side of retrieval.

Stage 1 (pre-retrieval, ~0.05 ms): compare the query vector against the
corpus topic centroids built by data/build_index.py. If the query is nowhere
near anything the corpus covers -- "who won the cricket match last night" against
an MS MARCO general-knowledge corpus -- we refuse without paying for the index
scan, the reranker, or generation. That is the cheap, decisive gate.

Stage 2 (post-retrieval, free): the cross-encoder has now actually read the
query against the best passages. Its top relevance score is a far better
signal than any pre-retrieval heuristic, and it costs nothing extra because
retrieval already computed it. Weak best-match means refuse rather than answer
from irrelevant context.

Two stages rather than one because they trade off differently: stage 1 is
cheap but coarse, stage 2 is accurate but only exists after the work is done.
"""
from typing import Optional, Tuple

import config


def check_pre_retrieval(centroid_similarity: float) -> Tuple[bool, Optional[str]]:
    """Return (is_off_topic, reason) from the nearest-centroid cosine."""
    if centroid_similarity < config.OFF_TOPIC_CENTROID_THRESHOLD:
        return True, (
            f"query is semantically distant from the indexed corpus "
            f"(nearest-topic cosine {centroid_similarity:.3f} < "
            f"{config.OFF_TOPIC_CENTROID_THRESHOLD})"
        )
    return False, None


def check_post_retrieval(retrieval) -> Tuple[bool, Optional[str]]:
    """Return (is_low_confidence, reason) from the reranked top score."""
    if not retrieval.chunks:
        return True, "retrieval returned no candidate passages"

    # Only meaningful on the cross-encoder scale. When the reranker was
    # skipped for latency we are looking at RRF scores instead, which have no
    # absolute meaning -- so we do not gate on them and say so.
    if not retrieval.reranked:
        return False, None

    top = retrieval.chunks[0].score
    if top < config.MIN_RELEVANCE_SCORE:
        return True, (
            f"best passage scored {top:.2f}, below the "
            f"{config.MIN_RELEVANCE_SCORE} relevance floor"
        )
    return False, None
