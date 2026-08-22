"""Pre-STT silence detection.

Runs before the STT network call because near-silent audio does not come back
as an empty transcript -- it tends to come back as a confidently hallucinated
one, which then flows through retrieval and generation as if it were a real
question. Rejecting it here costs ~1 ms and saves a wrong answer.

Only 16-bit PCM WAV can be inspected without a decoder dependency. The browser
client re-encodes its recording to exactly that before upload; anything else
returns "not silence" so the request still reaches STT rather than being
wrongly rejected. `probe()` reports which of those two happened, so a silently
skipped check is visible instead of invisible.
"""
import struct
import wave
from typing import Optional

import config


def probe(audio_file_path: str) -> dict:
    """Inspect the file. Returns rms/checked/reason without deciding."""
    try:
        with wave.open(audio_file_path, "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            n = w.getnframes()
            frames = w.readframes(n)
    except Exception as exc:  # noqa: BLE001 - not a WAV we can read
        return {"checked": False, "reason": f"undecodable ({type(exc).__name__})",
                "rms": None}

    if width != 2:
        return {"checked": False, "reason": f"{width * 8}-bit, expected 16-bit",
                "rms": None}

    count = len(frames) // 2
    if count == 0:
        return {"checked": True, "reason": "empty", "rms": 0.0,
                "duration_s": 0.0, "rate": rate}

    samples = struct.unpack("<%dh" % count, frames)
    # Mixed-down RMS is fine for a loudness gate; no need to split channels.
    rms = (sum(s * s for s in samples) / count) ** 0.5
    return {
        "checked": True,
        "reason": "",
        "rms": rms,
        "duration_s": round(n / rate, 3) if rate else None,
        "rate": rate,
        "channels": channels,
    }


def is_likely_silence(audio_file_path: str, threshold: Optional[int] = None) -> bool:
    """True when the clip is quiet enough to reject before calling STT.

    Fails open: if the audio cannot be inspected we return False so the
    request still reaches STT. Better to spend one API call than to refuse a
    question we simply could not measure.
    """
    threshold = config.SILENCE_RMS_THRESHOLD if threshold is None else threshold
    info = probe(audio_file_path)
    if not info["checked"]:
        return False
    return info["rms"] < threshold
