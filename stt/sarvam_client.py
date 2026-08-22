# stt/sarvam_client.py
"""Sarvam speech-to-text client.

Sarvam is used rather than a local Whisper because the target language is
Hindi and Sarvam's models are trained for Indic speech; the cost is that this
stage is an HTTPS round trip, which is why STT is measured separately from the
200 ms RAG-core budget.

Retries use exponential backoff and only cover transient failures. A 401 or a
400 will fail the same way three times in a row, so retrying them just
multiplies the user's wait.
"""
import mimetypes
import os
import time

import requests

import config
from harness.schemas import STTResult

# Status codes worth trying again: rate limiting and server-side faults.
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class STTError(RuntimeError):
    pass


class STTConfigError(STTError):
    """Raised for problems retrying cannot fix (missing/invalid key)."""


def transcribe(audio_file_path: str, language_code: str = None) -> STTResult:
    """Transcribe one audio file. Raises STTError on failure."""
    language_code = language_code or config.STT_LANGUAGE_CODE
    if not config.SARVAM_API_KEY:
        raise STTConfigError(
            "SARVAM_API_KEY is not set. Add it to .env, or use the text "
            "endpoint /api/ask which needs no STT key."
        )

    start = time.perf_counter()
    mime_type, _ = mimetypes.guess_type(audio_file_path)
    with open(audio_file_path, "rb") as f:
        try:
            resp = requests.post(
                config.STT_ENDPOINT,
                headers={"api-subscription-key": config.SARVAM_API_KEY},
                # Send the bare filename, not the full temp path.
                files={"file": (os.path.basename(audio_file_path), f,
                                mime_type or "audio/wav")},
                data={"language_code": language_code, "model": config.STT_MODEL},
                timeout=config.STT_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise STTError(f"STT request failed: {exc}") from exc

    if resp.status_code in (401, 403):
        raise STTConfigError(
            f"Sarvam rejected the API key (HTTP {resp.status_code}). "
            "Check SARVAM_API_KEY."
        )
    if resp.status_code >= 400:
        err = STTError(f"Sarvam returned HTTP {resp.status_code}: {resp.text[:200]}")
        err.status_code = resp.status_code
        raise err

    try:
        result = resp.json()
    except ValueError as exc:
        raise STTError(f"Sarvam returned non-JSON: {resp.text[:200]}") from exc

    transcript = result.get("transcript")
    if transcript is None:
        raise STTError(f"No transcript in Sarvam response: {result}")

    return STTResult(
        transcript=transcript.strip(),
        language=result.get("language_code", language_code),
        confidence=float(result.get("confidence", 1.0) or 1.0),
        duration_ms=(time.perf_counter() - start) * 1000,
    )


def transcribe_with_retry(
    audio_file_path: str, language_code: str = None, max_retries: int = None
) -> STTResult:
    """Transcribe with exponential backoff on transient failures only."""
    max_retries = config.STT_MAX_RETRIES if max_retries is None else max_retries
    last_err = None

    for attempt in range(max_retries + 1):
        try:
            return transcribe(audio_file_path, language_code)
        except STTConfigError:
            # Bad credentials will not fix themselves. Fail immediately.
            raise
        except STTError as exc:
            last_err = exc
            status = getattr(exc, "status_code", None)
            if status is not None and status not in _RETRYABLE_STATUS:
                raise
            if attempt < max_retries:
                time.sleep(0.4 * (2 ** attempt))

    raise STTError(f"STT failed after {max_retries} retries: {last_err}")
