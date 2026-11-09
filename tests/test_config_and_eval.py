from pathlib import Path

import pytest

from ragagent.agent import Agent
from ragagent.config import Settings
from ragagent.evaluate import evaluate, load_golden
from ragagent.factory import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_BACKEND", "AZURE")
    monkeypatch.setenv("RAG_TOP_K", "7")
    s = Settings.from_env()
    assert (s.backend, s.top_k) == ("azure", 7)


def test_azure_backend_without_env_fails_clearly() -> None:
    with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
        build_pipeline(Settings(backend="azure"))


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown RAG_BACKEND"):
        build_pipeline(Settings(backend="pinecone"))


def test_golden_set_eval_meets_floor(agent: Agent) -> None:
    items = load_golden(ROOT / "data" / "eval" / "golden.jsonl")
    assert len(items) >= 20
    report = evaluate(agent.pipeline, items)
    # Regression floor, set below the measured offline numbers reported in the README.
    assert report["metrics"]["hit@5"] >= 0.9
    assert report["metrics"]["answer_has_citation"] >= 0.9
