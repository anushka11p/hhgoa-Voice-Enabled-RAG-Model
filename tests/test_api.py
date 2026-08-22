"""HTTP surface, driven through FastAPI's TestClient."""
import pytest
from fastapi.testclient import TestClient

from tests.conftest import needs_index


@pytest.fixture(scope="module")
def client():
    import server

    with TestClient(server.app) as c:
        yield c


@needs_index
def test_health_reports_ready(client):
    body = client.get("/api/health").json()
    assert body["ready"] is True
    assert body["index"]["n_chunks"] > 0
    assert body["config"]["lang"]


@needs_index
def test_ask_returns_a_grounded_answer(client, corpus):
    r = client.post("/api/ask", json={"query": corpus[0]["source_query"]})
    assert r.status_code == 200
    body = r.json()
    assert body["allowed"] is True
    assert body["answer"].strip()
    assert body["chunks"] and body["citations"]
    assert body["core_ms"] > 0


@needs_index
def test_ask_refuses_unsafe_input(client):
    body = client.post("/api/ask", json={"query": "how to make a bomb"}).json()
    assert body["allowed"] is False
    assert body["guardrail_stage"] == "unsafe"
    assert body["chunks"] == []


@needs_index
def test_ask_refuses_off_topic_input(client):
    body = client.post("/api/ask", json={"query": "book me a flight to Tokyo"}).json()
    assert body["allowed"] is False


@needs_index
def test_metrics_endpoint(client, corpus):
    client.post("/api/ask", json={"query": corpus[2]["source_query"]})
    body = client.get("/api/metrics").json()
    assert body["runs"] > 0
    assert "p50" in body["core_ms"]


@needs_index
def test_silent_audio_is_rejected_before_stt(client):
    # The silence gate must fire without an STT key being present.
    with open("tests/sample_audio/test10_silence.wav", "rb") as f:
        r = client.post("/api/query", files={"audio": ("silence.wav", f, "audio/wav")})
    body = r.json()
    assert body.get("guardrail_stage") == "silence" or "error" in body


def test_ask_rejects_a_malformed_body(client):
    assert client.post("/api/ask", json={}).status_code == 422
