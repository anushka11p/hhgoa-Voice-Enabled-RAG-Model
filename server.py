from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import tempfile

from harness.orchestrator import run_pipeline, PipelineError
from harness.logging_utils import log_run
from harness.stubs import stub_retrieve, stub_generate, stub_guardrail

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/query")
async def query(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, stub_retrieve, stub_generate, stub_guardrail)
        log_run(result)
        return {
            "transcript": result.transcript,
            "answer": result.answer,
            "allowed": result.allowed,
            "stage_timings_ms": result.stage_timings_ms,
            "total_ms": result.total_ms,
        }
    except PipelineError as e:
        return {"error": str(e)}

@app.get("/")
def index():
    return FileResponse("frontend/index.html")

app.mount("/static", StaticFiles(directory="frontend"), name="static")