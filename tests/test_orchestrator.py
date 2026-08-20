import sys
sys.path.append(".")

from harness.orchestrator import run_pipeline
from harness.stubs import stub_retrieve, stub_generate, stub_guardrail

result = run_pipeline("tests/sample_audio/test1_diabetes.wav", stub_retrieve, stub_generate, stub_guardrail)
print(result)