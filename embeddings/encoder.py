"""Process-wide singleton encoder producing L2-normalized vectors.

Two reasons this exists alongside embeddings/embed.py:

1. Serving needs the model loaded exactly once, at boot, not per request.
   Re-instantiating SentenceTransformer inside a request costs seconds.
2. Vectors are L2-normalized here. With unit vectors, the FAISS IndexFlatL2
   distance and cosine similarity are monotonically related --
   cos = 1 - (d^2 / 2) -- so we get calibrated cosine scores (which the
   off-topic guardrail thresholds against) out of an exact L2 index for free.
"""
import threading
from typing import List

import numpy as np

import config

_model = None
_lock = threading.Lock()


def get_model():
    """Load the bi-encoder once, thread-safely."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import torch
                from sentence_transformers import SentenceTransformer

                # Matches the env vars config.py already pinned.
                torch.set_num_threads(config.TORCH_NUM_THREADS)

                print(f"[encoder] loading {config.EMBED_MODEL} ...")
                _model = SentenceTransformer(config.EMBED_MODEL)
                # Renamed in sentence-transformers 6.x; support both so the
                # pinned range in requirements.txt stays wide.
                get_dim = getattr(
                    _model, "get_embedding_dimension", None
                ) or _model.get_sentence_embedding_dimension
                dim = get_dim()
                if dim != config.VECTOR_DIM:
                    raise ValueError(
                        f"{config.EMBED_MODEL} emits {dim}-dim vectors but "
                        f"VECTOR_DIM is {config.VECTOR_DIM}. Set VECTOR_DIM to "
                        f"{dim} and rebuild the index."
                    )
                print(f"[encoder] ready (dim={dim})")
    return _model


def encode_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """Encode a batch of documents. Returns (n, dim) float32, unit-normalized."""
    if not texts:
        return np.zeros((0, config.VECTOR_DIM), dtype=np.float32)
    vecs = get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    """Encode a single query. Returns (dim,) float32, unit-normalized."""
    return encode_texts([text])[0]


def l2_to_cosine(distance: float) -> float:
    """Convert a FAISS L2 distance between unit vectors into cosine similarity."""
    return float(1.0 - (distance / 2.0))


def warmup() -> None:
    """Run one tiny encode so the first real request doesn't pay lazy-init cost."""
    encode_texts(["warmup"])
