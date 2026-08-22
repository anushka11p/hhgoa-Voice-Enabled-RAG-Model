"""Shared pytest fixtures.

Adds the repo root to sys.path so tests can run from anywhere, and provides a
session-scoped retrieval engine -- loading the models per-test would make the
suite take minutes instead of seconds.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def index_exists() -> bool:
    return (config.INDEX_DIR / f"{config.INDEX_NAME}.bin").exists()


needs_index = pytest.mark.skipif(
    not index_exists(),
    reason="index not built -- run `python -m data.build_corpus && python -m data.build_index`",
)

needs_stt_key = pytest.mark.skipif(
    not config.SARVAM_API_KEY, reason="SARVAM_API_KEY not set"
)


@pytest.fixture(scope="session")
def engine():
    from retrieval.pipeline import get_engine, warmup

    eng = get_engine()
    warmup(eng)
    return eng


@pytest.fixture(scope="session")
def corpus():
    import json

    return json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
