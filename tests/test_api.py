import os

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.rag import pipeline


class _FakeChain:
    """Stand-in for the Groq chain so unit tests don't hit the network."""

    def invoke(self, _inputs):
        return "Merge them with `{**a, **b}` [1]."


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(pipeline, "_chain", lambda: _FakeChain())
    routes._cache.clear()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["index_loaded"] is True
    assert body["num_vectors"] > 0


def test_ask_happy_path(client):
    r = client.post("/ask", json={"question": "how do I reverse a list in python"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["url"].startswith("https://stackoverflow.com/")


def test_empty_question_is_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_missing_field_is_422(client):
    assert client.post("/ask", json={}).status_code == 422


def test_off_topic_refuses(client):
    r = client.post("/ask", json={"question": "what is the capital of France"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == []
    assert "grounded answer" in body["answer"].lower()


def test_second_identical_request_is_cached(client):
    q = {"question": "how do I reverse a list in python"}
    first = client.post("/ask", json=q).json()
    assert "how do i reverse a list in python" in routes._cache
    second = client.post("/ask", json=q).json()
    # cache returns the stored result verbatim, including the original latency
    assert second["latency_ms"] == first["latency_ms"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION"), reason="set RUN_INTEGRATION=1 to hit real Groq"
)
def test_ask_real_llm():
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "how do I reverse a list in python"})
        assert r.status_code == 200
        assert r.json()["answer"]
