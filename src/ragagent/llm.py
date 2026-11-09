"""Answer generators: an offline extractive stub (default) and Azure OpenAI chat."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ragagent.vectorstore import SearchResult

NO_ANSWER = "I couldn't find that in the policy documents."

SYSTEM_PROMPT = (
    "You answer employee questions using only the numbered context passages. "
    "Cite every fact with its passage number in square brackets, like [1]. "
    f"If the context does not contain the answer, reply exactly: {NO_ANSWER}"
)


class AnswerGenerator(Protocol):
    def generate(self, question: str, results: Sequence[SearchResult]) -> str: ...


def _terms(text: str) -> set[str]:
    """Lower-case content words with a crude plural strip ("days" -> "day")."""
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words} - ENGLISH_STOP_WORDS


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


class ExtractiveGenerator:
    """Offline "LLM" stub: returns the retrieved sentences that best overlap the question.

    It does not paraphrase or reason; it only selects sentences, each tagged with
    the number of the passage it came from, so every answer is grounded and cited.
    """

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        q_terms = _terms(question)
        scored: list[tuple[float, int, int, str]] = []
        for passage_no, result in enumerate(results, start=1):
            for position, sentence in enumerate(split_sentences(result.chunk.text)):
                overlap = len(q_terms & _terms(sentence))
                if overlap:
                    # word overlap first, retrieval score breaks ties
                    scored.append((overlap + result.score, passage_no, position, sentence))
        if not scored:
            return NO_ANSWER
        best = sorted(scored, key=lambda s: -s[0])[: self.max_sentences]
        best.sort(key=lambda s: (s[1], s[2]))  # keep reading order
        return " ".join(f"{sentence} [{passage_no}]" for _, passage_no, _, sentence in best)


def format_context(results: Sequence[SearchResult]) -> str:
    return "\n\n".join(
        f"[{i}] {r.chunk.title} / {r.chunk.section}\n{r.chunk.text}"
        for i, r in enumerate(results, start=1)
    )


class AzureChatGenerator:
    """Grounded answer from an Azure OpenAI chat deployment (e.g. gpt-4o-mini)."""

    def __init__(self, client: Any, deployment: str) -> None:
        self._client = client  # an openai.AzureOpenAI instance
        self._deployment = deployment

    def generate(self, question: str, results: Sequence[SearchResult]) -> str:
        response = self._client.chat.completions.create(
            model=self._deployment,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{format_context(results)}\n\nQuestion: {question}",
                },
            ],
        )
        return response.choices[0].message.content or NO_ANSWER
