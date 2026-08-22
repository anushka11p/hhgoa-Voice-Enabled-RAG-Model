# Container image for the Voice-Enabled RAG demo.
#
# Two deliberate choices:
#  * Models are baked in at build time. They are ~950 MB of weights; pulling
#    them on first request would make a cold start take minutes and would fail
#    entirely on a host without outbound Hugging Face access.
#  * The index bundle is committed to the repo, so the container does not
#    re-embed 1.5k chunks on every boot.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf \
    TORCH_NUM_THREADS=1

WORKDIR /app

# Torch's CPU wheels need no build toolchain, but faiss does want libgomp.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch: the default wheel drags in CUDA and inflates the image by GBs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the bi-encoder and cross-encoder into the image.
RUN python -c "\
import config; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(config.EMBED_MODEL); \
CrossEncoder(config.RERANK_MODEL, max_length=256); \
print('models cached')"

# Hugging Face Spaces routes to 7860; override with PORT elsewhere.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
