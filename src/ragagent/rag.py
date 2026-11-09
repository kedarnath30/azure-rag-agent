"""The RAG pipeline: index chunks, retrieve top-k, generate a cited answer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ragagent.documents import Chunk
from ragagent.embeddings import Embedder
from ragagent.llm import NO_ANSWER, AnswerGenerator
from ragagent.vectorstore import SearchResult, VectorStore


@dataclass(frozen=True)
class Citation:
    number: int  # the [n] used in the answer text
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: AnswerGenerator,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    def index(self, chunks: Sequence[Chunk]) -> None:
        texts = [c.search_text for c in chunks]
        self.embedder.fit(texts)
        self.store.add(chunks, self.embedder.embed(texts))

    def retrieve(self, query: str, k: int | None = None) -> list[SearchResult]:
        vector = self.embedder.embed([query])[0]
        results = self.store.search(vector, k or self.top_k)
        return [r for r in results if r.score >= self.min_score]

    def ask(self, question: str, k: int | None = None) -> Answer:
        results = self.retrieve(question, k)
        if not results:
            return Answer(answer=NO_ANSWER, citations=[])
        text = self.generator.generate(question, results)
        citations = [
            Citation(
                number=i,
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                section=r.chunk.section,
                score=round(r.score, 4),
                snippet=r.chunk.text[:200],
            )
            for i, r in enumerate(results, start=1)
        ]
        return Answer(answer=text, citations=citations)
