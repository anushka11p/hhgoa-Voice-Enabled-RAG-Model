import wave
import struct

def is_likely_silence(audio_file_path: str, threshold: int = 300) -> bool:
    try:
        with wave.open(audio_file_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            sample_width = w.getsampwidth()
            if sample_width != 2:
                return False  # only handling standard 16-bit audio here

            count = len(frames) // 2
            samples = struct.unpack("<%dh" % count, frames)
            if not samples:
                return True

            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / count) ** 0.5
            return rms < threshold
    except Exception:
        return False