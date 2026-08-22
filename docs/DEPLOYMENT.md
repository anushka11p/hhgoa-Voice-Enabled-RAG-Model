# Deployment

## Resource requirements — read this first

This stack holds two transformer models plus a FAISS index resident in memory:

| Component | Approx. size |
|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (bi-encoder) | ~470 MB |
| `mmarco-mMiniLMv2-L12-H384-v1` (cross-encoder) | ~470 MB |
| FAISS index + BM25 + metadata (committed) | ~7 MB |
| Torch runtime + process overhead | ~300 MB |

**Plan for ~1.5 GB RAM.** A 512 MB instance will OOM on startup — this rules
out Render's free tier and most "hobby" plans.

---

## Recommended: Hugging Face Spaces (free, fits comfortably)

**Docker Spaces are free.** Docker is a standard SDK option alongside Gradio
and Streamlit when creating a Space -- it does not require PRO. You also do
not need Docker installed locally or a Docker Hub account; Hugging Face builds
the image on their infrastructure. What costs money is *hardware upgrades*
(GPUs, more RAM), and this app does not need them: CPU basic is 2 vCPU / 16 GB,
which is comfortably above the ~1.5 GB this stack uses.

CPU Basic gives 2 vCPU / 16 GB RAM at no cost, and caches Hugging Face model
downloads — the best fit for this workload.

The YAML frontmatter at the top of `README.md` is what configures the Space
(`sdk: docker`, `app_port: 7860`). Spaces reads it on every push -- if it is
removed, the Space will not know how to build.

1. Create a Space at <https://huggingface.co/new-space>
   - **SDK:** Docker
   - **Hardware:** CPU basic (free)
2. Push this repo to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```
3. Add `SARVAM_API_KEY` under **Settings → Variables and secrets**
   (as a *secret*, not a variable).
4. The included `Dockerfile` listens on `7860`, which is what Spaces expects.

First build takes ~10 minutes (it bakes the models into the image so cold
starts are fast). After that, boot is a few seconds.

**Voice input needs HTTPS** — browsers block `getUserMedia` on plain HTTP
except on `localhost`. Spaces serves HTTPS, so the mic works. The text
endpoint works either way.

---

## Alternative: Render

`render.yaml` is included, but note the plan:

```yaml
plan: standard   # NOT free — free is 512 MB and will OOM
```

1. New → Blueprint → point at this repo.
2. Set `SARVAM_API_KEY` (and optionally `ANTHROPIC_API_KEY`) in the dashboard.
   They are marked `sync: false` so they are never committed.
3. Health check is `/api/health`, which reports readiness and the index
   manifest.

---

## Any container host

```bash
docker build -t voice-rag .
docker run -p 8000:8000 -e PORT=8000 -e SARVAM_API_KEY=... voice-rag
```

The image installs CPU-only torch (the default wheel pulls CUDA and adds
gigabytes) and pre-downloads both models at build time, so the container needs
no outbound Hugging Face access at runtime.

---

## Verifying a deployment

```bash
curl -s https://<your-host>/api/health | python3 -m json.tool
```

Expect `"ready": true` and a non-zero `index.n_chunks`. If `ready` is false,
the `error` field says exactly what failed — the app boots and reports rather
than crash-looping, so you can diagnose from the health endpoint.

Then confirm the pipeline answers and refuses:

```bash
curl -s -X POST https://<your-host>/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"मायस्थेनिया ग्रेविस का इलाज क्या है?"}'

curl -s -X POST https://<your-host>/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"who won the cricket world cup in 2011"}'
```

The first should return `"allowed": true` with citations; the second
`"allowed": false` with `"guardrail_stage": "off_topic"`.

---

## Pre-submission checklist

- [ ] Fresh clone, fresh venv, `pip install -r requirements.txt`, `pytest -q`
- [ ] `uvicorn server:app` serves from the committed index with no rebuild
- [ ] `/api/health` reports `ready: true` on the **deployed** URL, not just locally
- [ ] Mic works on the deployed HTTPS URL
- [ ] `SARVAM_API_KEY` set in the host's secret store; `.env` is gitignored
- [ ] Live link added to the README header
- [ ] Demo video recorded against the deployed version
- [ ] `#RAGInGoa` on every promo post; at least one Instagram account public
