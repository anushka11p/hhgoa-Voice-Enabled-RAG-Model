"""Sarvam STT against the committed sample audio.

Skipped when SARVAM_API_KEY is absent, so the suite still runs green on a
clean checkout. Originally a manual script; converted to pytest so CI can
collect it without firing a network call at import time.
"""
import pytest

from stt.sarvam_client import STTConfigError, transcribe, transcribe_with_retry
from tests.conftest import needs_stt_key


@needs_stt_key
def test_transcribes_hindi_sample():
    result = transcribe("tests/sample_audio/test1_diabetes.wav")
    assert result.transcript.strip()
    assert result.duration_ms > 0


@needs_stt_key
def test_retry_wrapper_returns_the_same_shape():
    result = transcribe_with_retry("tests/sample_audio/test1_diabetes.wav")
    assert result.transcript.strip()


def test_missing_key_fails_fast_without_retrying(monkeypatch):
    # A credential problem must raise immediately -- retrying it three times
    # only multiplies the user's wait.
    import config

    monkeypatch.setattr(config, "SARVAM_API_KEY", None)
    with pytest.raises(STTConfigError):
        transcribe_with_retry("tests/sample_audio/test1_diabetes.wav")
