"""Harness-level tests for the voice entry point and the silence gate."""
import pytest

from stt.audio_check import is_likely_silence
from tests.conftest import needs_index, needs_stt_key


def test_silence_gate_flags_silent_audio():
    assert is_likely_silence("tests/sample_audio/test10_silence.wav")


def test_silence_gate_passes_real_speech():
    assert not is_likely_silence("tests/sample_audio/test1_diabetes.wav")


def test_silence_gate_survives_a_bad_path():
    # Must not raise -- a decode failure should fall through to STT, not 500.
    assert is_likely_silence("tests/sample_audio/does_not_exist.wav") is False


@needs_index
@needs_stt_key
def test_full_voice_pipeline():
    from harness.orchestrator import run_from_audio

    result = run_from_audio("tests/sample_audio/test1_diabetes.wav")
    assert result.transcript.strip()
    assert "stt_ms" in result.stage_timings_ms
    # STT is measured but excluded from the core budget.
    assert result.core_ms < result.total_ms


@needs_index
@needs_stt_key
def test_legacy_stub_pipeline_still_runs():
    from harness.orchestrator import run_pipeline
    from harness.stubs import stub_generate, stub_guardrail, stub_retrieve

    result = run_pipeline(
        "tests/sample_audio/test1_diabetes.wav",
        stub_retrieve,
        stub_generate,
        stub_guardrail,
    )
    assert result.allowed
