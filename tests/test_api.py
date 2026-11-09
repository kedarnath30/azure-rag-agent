from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ragagent.api import create_app
from ragagent.config import Settings


@pytest.fixture(scope="module")
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_ask_returns_answer_and_citations(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the daily meal limit?", "top_k": 3})
    body = resp.json()
    assert resp.status_code == 200
    assert "$75" in body["answer"]
    assert body["citations"][0]["doc_id"] == "expense-reimbursement"
    assert len(body["citations"]) <= 3


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 50}).status_code == 422


def test_agent_endpoint_reports_tool_steps(client: TestClient) -> None:
    resp = client.post("/agent", json={"question": "How much vacation do I get with 1 year?"})
    body = resp.json()
    assert body["steps"][0]["tool"] == "calculate_pto"
    assert "15 PTO days" in body["answer"]
