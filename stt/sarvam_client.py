# stt/sarvam_client.py
import mimetypes
import requests
import time
import os
from dotenv import load_dotenv
from harness.schemas import STTResult

load_dotenv()  # reads .env file so you don't have to export manually every time

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
STT_ENDPOINT = "https://api.sarvam.ai/speech-to-text"

def transcribe(audio_file_path: str, language_code: str = "hi-IN") -> STTResult:
    start = time.perf_counter()
    with open(audio_file_path, "rb") as f:
        mime_type, _ = mimetypes.guess_type(audio_file_path)
        files = {"file": (audio_file_path, f, mime_type or "audio/wav")}
        headers = {"api-subscription-key": SARVAM_API_KEY}
        data = {"language_code": language_code}
        resp = requests.post(STT_ENDPOINT, headers=headers, files=files, data=data, timeout=10)
    print("STATUS:", resp.status_code)
    print("BODY:", resp.text)
    resp.raise_for_status()
    result = resp.json()
    elapsed = (time.perf_counter() - start) * 1000

    return STTResult(
        transcript=result["transcript"],
        language=language_code,
        confidence=result.get("confidence", 1.0),
        duration_ms=elapsed,
    )
def transcribe_with_retry(audio_file_path, language_code="hi-IN", max_retries=2):
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return transcribe(audio_file_path, language_code)
        except Exception as e:
            last_err = e
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"STT failed after {max_retries} retries: {last_err}")