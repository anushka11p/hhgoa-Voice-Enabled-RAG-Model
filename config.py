"""Central configuration for the Voice-Enabled RAG pipeline.

Everything tunable lives here so the pipeline can be re-pointed at a different
language, model, or latency budget without touching module code. Values are
read from the environment first (so deployments can override via env vars) and
fall back to the defaults below.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Pin the math-library thread pools BEFORE torch/numpy get imported anywhere.
# These models run one short query at a time, not big batches: extra threads
# only add scheduling contention. Measured on this box, single-threaded cut
# query-embedding p100 from ~48 ms to ~12 ms and made rerank latency far more
# predictable, which is what a percentile budget actually cares about.
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(TORCH_NUM_THREADS))


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------
# "hi" runs the whole stack in Hindi (matches the Sarvam hi-IN STT and the
# ai4bharat Hindi corpus). "en" swaps in the smaller English-only models.
LANG = _env("RAG_LANG", "hi").lower()

STT_LANGUAGE_CODE = _env("STT_LANGUAGE_CODE", "hi-IN" if LANG == "hi" else "en-IN")

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Both defaults emit 384-dim vectors, which is what the FAISS store expects.
_DEFAULT_EMBED = {
    "hi": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "en": "sentence-transformers/all-MiniLM-L6-v2",
}
_DEFAULT_RERANK = {
    "hi": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "en": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

EMBED_MODEL = _env("EMBED_MODEL", _DEFAULT_EMBED.get(LANG, _DEFAULT_EMBED["en"]))
RERANK_MODEL = _env("RERANK_MODEL", _DEFAULT_RERANK.get(LANG, _DEFAULT_RERANK["en"]))
VECTOR_DIM = _env_int("VECTOR_DIM", 384)

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# IndicMSMARCO is the ~1k-row-per-language companion to MSMARCO-XI. It carries
# the same query -> passage pairing, so it doubles as our retrieval eval set and
# latency benchmark set, and it downloads in ~1 MB instead of ~3.7 GB.
DATASET_REPO = _env("DATASET_REPO", "ai4bharat/IndicMSMARCO")
DATASET_FILE = _env("DATASET_FILE", f"{LANG}/train-00000-of-00001.parquet")
MAX_CORPUS_ROWS = _env_int("MAX_CORPUS_ROWS", 1000)

DATA_DIR = BASE_DIR / "data" / "processed"
CORPUS_PATH = DATA_DIR / "cleaned_corpus.json"
INDEX_DIR = BASE_DIR / "data" / "index"
INDEX_NAME = _env("INDEX_NAME", "voice_rag")

# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
# The strategy actually indexed for serving. All four are implemented and
# benchmarked in analytics/compare_chunking.py; this picks the winner.
ACTIVE_CHUNK_STRATEGY = _env("ACTIVE_CHUNK_STRATEGY", "recursive")
CHUNK_SIZE = _env_int("CHUNK_SIZE", 400)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
DENSE_TOP_K = _env_int("DENSE_TOP_K", 20)
BM25_TOP_K = _env_int("BM25_TOP_K", 20)
RRF_K = _env_int("RRF_K", 60)
RRF_TOP_N = _env_int("RRF_TOP_N", 8)       # candidates handed to the reranker
FINAL_TOP_K = _env_int("FINAL_TOP_K", 4)   # chunks handed to generation

# --------------------------------------------------------------------------
# Latency budgets (milliseconds)
# --------------------------------------------------------------------------
# The 200 ms target covers the RAG core: retrieval + generation + guardrails.
# Speech-to-text is a network round trip to Sarvam's API and is measured and
# reported separately -- see docs/ARCHITECTURE.md for the reasoning.
CORE_BUDGET_MS = _env_float("CORE_BUDGET_MS", 200.0)
BUDGET_EMBED_MS = _env_float("BUDGET_EMBED_MS", 40.0)
BUDGET_RETRIEVAL_MS = _env_float("BUDGET_RETRIEVAL_MS", 120.0)
BUDGET_RERANK_MS = _env_float("BUDGET_RERANK_MS", 80.0)
BUDGET_GENERATION_MS = _env_float("BUDGET_GENERATION_MS", 40.0)
BUDGET_GUARDRAIL_MS = _env_float("BUDGET_GUARDRAIL_MS", 25.0)

# The reranker's cost is ~linear in candidate count, so rather than skipping it
# outright under deadline pressure we shrink how many candidates it scores.
# Seed estimate for one candidate; the engine replaces this with a live EWMA.
RERANK_MS_PER_CANDIDATE_SEED = _env_float("RERANK_MS_PER_CANDIDATE_SEED", 9.0)
# The EWMA is clamped to this range. Without a ceiling, one query that happened
# to run during a CPU spike inflates the estimate, which shrinks the next
# query's rerank width, which keeps the estimate high -- the pipeline would
# quietly stop reranking long after the machine recovered.
RERANK_MS_PER_CANDIDATE_MIN = _env_float("RERANK_MS_PER_CANDIDATE_MIN", 2.0)
RERANK_MS_PER_CANDIDATE_MAX = _env_float("RERANK_MS_PER_CANDIDATE_MAX", 20.0)
# Held back from the budget for generation + post guardrails + response build.
POST_RERANK_RESERVE_MS = _env_float("POST_RERANK_RESERVE_MS", 12.0)
# Below this many affordable candidates, skip reranking entirely -- scoring one
# or two candidates does not reorder anything worth the latency.
MIN_RERANK_CANDIDATES = _env_int("MIN_RERANK_CANDIDATES", 3)

# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
# "extractive" is the deterministic, sub-millisecond default that keeps the
# pipeline inside the 200 ms budget and is grounded by construction.
# "llm" calls Claude for fluent synthesis (slower, needs ANTHROPIC_API_KEY).
# "auto" uses the LLM only when a key is present.
GENERATION_MODE = _env("GENERATION_MODE", "extractive").lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "claude-sonnet-4-5")
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 300)
LLM_TIMEOUT_S = _env_float("LLM_TIMEOUT_S", 12.0)

REFUSAL_TEXT = _env(
    "REFUSAL_TEXT",
    "मेरे पास इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
    if LANG == "hi"
    else "I don't have enough information to answer that.",
)

# --------------------------------------------------------------------------
# Guardrail thresholds
# --------------------------------------------------------------------------
# Both thresholds were chosen by analytics/calibrate_guardrails.py, which
# scores 120 real dataset queries against 39 deliberately out-of-corpus ones
# and grid-searches the two-signal rule. At this operating point the gate
# refuses 8.3% of genuine questions and catches 69.2% of off-topic ones.
# Neither signal separates the populations alone -- that is why there are two,
# and why the grounding check downstream is a third net.
#
# Cosine to the nearest corpus centroid, checked BEFORE retrieval runs.
OFF_TOPIC_CENTROID_THRESHOLD = _env_float("OFF_TOPIC_CENTROID_THRESHOLD", 0.34)
# Cross-encoder relevance of the best chunk, checked AFTER retrieval.
MIN_RELEVANCE_SCORE = _env_float("MIN_RELEVANCE_SCORE", -4.5)
# Fraction of the answer's content tokens that must appear in the cited chunks.
GROUNDING_OVERLAP_THRESHOLD = _env_float("GROUNDING_OVERLAP_THRESHOLD", 0.55)
MIN_QUERY_CHARS = _env_int("MIN_QUERY_CHARS", 3)

# --------------------------------------------------------------------------
# STT
# --------------------------------------------------------------------------
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
STT_ENDPOINT = _env("STT_ENDPOINT", "https://api.sarvam.ai/speech-to-text")
STT_MODEL = _env("STT_MODEL", "saarika:v2")
STT_TIMEOUT_S = _env_float("STT_TIMEOUT_S", 15.0)
STT_MAX_RETRIES = _env_int("STT_MAX_RETRIES", 2)
SILENCE_RMS_THRESHOLD = _env_int("SILENCE_RMS_THRESHOLD", 300)

# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
ANALYTICS_DIR = BASE_DIR / "analytics"
LATENCY_LOG_PATH = ANALYTICS_DIR / "latency_log.jsonl"
BENCHMARK_RESULTS_PATH = ANALYTICS_DIR / "benchmark_results.json"


def summary() -> dict:
    """Config snapshot, surfaced by the /api/health endpoint."""
    return {
        "lang": LANG,
        "stt_language_code": STT_LANGUAGE_CODE,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "dataset": f"{DATASET_REPO}:{DATASET_FILE}",
        "chunk_strategy": ACTIVE_CHUNK_STRATEGY,
        "generation_mode": GENERATION_MODE,
        "core_budget_ms": CORE_BUDGET_MS,
        "stt_key_present": bool(SARVAM_API_KEY),
        "llm_key_present": bool(ANTHROPIC_API_KEY),
    }
