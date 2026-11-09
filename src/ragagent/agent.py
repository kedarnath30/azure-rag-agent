"""A small tool-using agent with plain-Python function calling.

Loop: the planner picks a tool call or a final answer; we run the tool,
record the step, and ask the planner again, up to `max_steps` times.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ragagent.rag import Citation, RAGPipeline
from ragagent.tools import TOOL_SCHEMAS, calculate_pto


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = "call_0"


@dataclass(frozen=True)
class FinalAnswer:
    text: str


@dataclass
class Step:
    call: ToolCall
    result: dict[str, Any]


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)


class Planner(Protocol):
    def next_action(
        self, question: str, steps: list[Step], tools: list[dict[str, Any]]
    ) -> ToolCall | FinalAnswer: ...


_PTO_WORDS = re.compile(r"\b(pto|vacation|paid time off|time off)\b", re.I)
_YEARS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\b", re.I)
_MONTHS = re.compile(r"(\d+)\s*months?\b", re.I)
_USED = re.compile(r"(?:used|taken|took)\s+(\d+(?:\.\d+)?)", re.I)


class RuleBasedPlanner:
    """Offline planner: regex routing instead of an LLM, so it is deterministic.

    A PTO question that mentions years of service goes to calculate_pto;
    everything else goes to search_docs. After one tool call it answers.
    """

    def next_action(
        self, question: str, steps: list[Step], tools: list[dict[str, Any]]
    ) -> ToolCall | FinalAnswer:
        if not steps:
            years = _YEARS.search(question)
            if _PTO_WORDS.search(question) and years:
                args: dict[str, Any] = {"years_of_service": float(years.group(1))}
                if months := _MONTHS.search(question):
                    args["months_worked_this_year"] = int(months.group(1))
                if used := _USED.search(question):
                    args["days_used"] = float(used.group(1))
                return ToolCall("calculate_pto", args)
            return ToolCall("search_docs", {"query": question})

        last = steps[-1]
        if last.call.name == "calculate_pto":
            r = last.result
            if "error" in r:
                return FinalAnswer(f"I couldn't calculate PTO: {r['error']}")
            return FinalAnswer(
                f"With {r['years_of_service']:g} years of service you earn {r['annual_days']} "
                f"PTO days per year. For {r['months_worked_this_year']} months worked that is "
                f"{r['accrued_days']:g} days; after {r['days_used']:g} used, "
                f"{r['remaining_days']:g} remain (up to {r['max_carryover_days']:g} can carry "
                "over). Source: PTO policy."
            )
        return FinalAnswer(str(last.result.get("answer", "")))


class AzureToolPlanner:
    """Planner backed by Azure OpenAI chat completions with `tools=`."""

    SYSTEM = (
        "You are an HR assistant for Brightwater Analytics. Use calculate_pto for PTO "
        "arithmetic and search_docs for policy questions. Keep the [n] citations that "
        "search_docs returns. Never invent policy details."
    )

    def __init__(self, client: Any, deployment: str) -> None:
        self._client = client
        self._deployment = deployment

    def _messages(self, question: str, steps: list[Step]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": question},
        ]
        for step in steps:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": step.call.call_id,
                            "type": "function",
                            "function": {
                                "name": step.call.name,
                                "arguments": json.dumps(step.call.arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": step.call.call_id,
                    "content": json.dumps(step.result),
                }
            )
        return messages

    def next_action(
        self, question: str, steps: list[Step], tools: list[dict[str, Any]]
    ) -> ToolCall | FinalAnswer:
        response = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            messages=self._messages(question, steps),
            tools=tools,
        )
        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]  # one tool at a time keeps the loop simple
            return ToolCall(call.function.name, json.loads(call.function.arguments), call.id)
        return FinalAnswer(message.content or "")


class Agent:
    def __init__(self, pipeline: RAGPipeline, planner: Planner, max_steps: int = 3) -> None:
        self.pipeline = pipeline
        self.planner = planner
        self.max_steps = max_steps
        self._citations: list[Citation] = []
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {
            "calculate_pto": calculate_pto,
            "search_docs": self._search_docs,
        }

    def _search_docs(self, query: str) -> dict[str, Any]:
        answer = self.pipeline.ask(query)
        self._citations = answer.citations
        return {"answer": answer.answer, "citations": [asdict(c) for c in answer.citations]}

    def _run_tool(self, call: ToolCall) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            return {"error": f"unknown tool: {call.name}"}
        try:
            return tool(**call.arguments)
        except (TypeError, ValueError) as exc:  # bad arguments from the planner
            return {"error": str(exc)}

    def run(self, question: str) -> AgentResult:
        self._citations = []
        steps: list[Step] = []
        for _ in range(self.max_steps):
            action = self.planner.next_action(question, steps, TOOL_SCHEMAS)
            if isinstance(action, FinalAnswer):
                return AgentResult(action.text, steps, self._citations)
            steps.append(Step(action, self._run_tool(action)))
        return AgentResult("Stopped: reached the step limit.", steps, self._citations)
