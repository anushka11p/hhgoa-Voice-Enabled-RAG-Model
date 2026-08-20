# stt/sarvam_client.py
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
        files = {"file": f}
        headers = {"api-subscription-key": SARVAM_API_KEY}
        data = {"language_code": language_code}
        resp = requests.post(STT_ENDPOINT, headers=headers, files=files, data=data, timeout=10)
        resp = requests.post(STT_ENDPOINT, headers=headers, files=files, data=data, timeout=10)
        print("STATUS:", resp.status_code)
        print("BODY:", resp.text)
        resp.raise_for_status()
        resp.raise_for_status()
    result = resp.json()
    elapsed = (time.perf_counter() - start) * 1000

    return STTResult(
        transcript=result["transcript"],
        language=language_code,
        confidence=result.get("confidence", 1.0),
        duration_ms=elapsed,
    )