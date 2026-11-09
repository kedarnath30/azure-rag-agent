"""Azure adapters tested against fake clients. No network; this checks our glue code only."""

from types import SimpleNamespace
from typing import Any

import numpy as np

from ragagent.agent import AzureToolPlanner, FinalAnswer, Step, ToolCall
from ragagent.documents import Chunk
from ragagent.embeddings import AzureOpenAIEmbedder
from ragagent.llm import AzureChatGenerator
from ragagent.tools import TOOL_SCHEMAS
from ragagent.vectorstore import SearchResult


class FakeOpenAI:
    def __init__(self, message: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.embeddings = SimpleNamespace(create=self._embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self._message = message

    def _embed(self, model: str, input: list[str]) -> Any:
        return SimpleNamespace(data=[SimpleNamespace(embedding=[3.0, 4.0]) for _ in input])

    def _chat(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


def test_azure_embedder_normalises_vectors() -> None:
    vectors = AzureOpenAIEmbedder(FakeOpenAI(None), "emb").embed(["a", "b"])
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_azure_generator_sends_numbered_context() -> None:
    client = FakeOpenAI(SimpleNamespace(content="Answer [1]", tool_calls=None))
    chunk = Chunk("pto-policy-0", "pto-policy", "PTO", "Carryover", "Up to 5 days.")
    text = AzureChatGenerator(client, "chat").generate("carryover?", [SearchResult(chunk, 0.9)])
    assert text == "Answer [1]"
    user_msg = client.calls[0]["messages"][1]["content"]
    assert "[1] PTO / Carryover" in user_msg


def test_azure_planner_parses_tool_call_and_replays_history() -> None:
    fn = SimpleNamespace(name="calculate_pto", arguments='{"years_of_service": 3}')
    message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=fn)])
    client = FakeOpenAI(message)
    planner = AzureToolPlanner(client, "chat")
    action = planner.next_action("pto for 3 years?", [], TOOL_SCHEMAS)
    assert action == ToolCall("calculate_pto", {"years_of_service": 3}, "c1")

    client._message = SimpleNamespace(content="You get 20 days.", tool_calls=None)
    final = planner.next_action("pto for 3 years?", [Step(action, {"annual_days": 20})], [])
    assert final == FinalAnswer("You get 20 days.")
    roles = [m["role"] for m in client.calls[1]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
