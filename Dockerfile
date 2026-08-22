# Container image for the Voice-Enabled RAG demo.
#
# Targets Hugging Face Spaces (Docker SDK, CPU basic) but works on any
# container host. Three deliberate choices:
#
#  * Runs as UID 1000, not root. Spaces runs containers as user 1000, so
#    anything written by root at build time is unreadable at runtime. Every
#    COPY sets --chown accordingly.
#  * Models are baked in at build time (~950 MB of weights). Pulling them on
#    first request would make a cold start take minutes and would fail on a
#    host without outbound Hugging Face access.
#  * The index bundle is committed to the repo, so the container never
#    re-embeds the corpus on boot.
FROM python:3.11-slim

# faiss needs libgomp; nothing else needs a build toolchain (torch ships wheels).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs as user 1000 -- create it before writing anything.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Model cache must live somewhere user 1000 can write.
    HF_HOME=/home/user/.cache/huggingface \
    # Single-threaded inference is measurably faster and far more predictable
    # for one-query-at-a-time serving. See docs/ARCHITECTURE.md.
    TORCH_NUM_THREADS=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
# CPU-only torch: the default wheel pulls CUDA and adds gigabytes.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Pre-download the bi-encoder and cross-encoder into the image layer.
RUN python -c "\
import config; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(config.EMBED_MODEL); \
CrossEncoder(config.RERANK_MODEL, max_length=256); \
print('models cached')"

# Spaces routes to 7860 (matches app_port in the README frontmatter).
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
