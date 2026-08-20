import sys
sys.path.append(".")  # so it can find stt/ and harness/ from repo root

from stt.sarvam_client import transcribe

result = transcribe("tests/sample_audio/test1_diabetes.wav")
print(result)