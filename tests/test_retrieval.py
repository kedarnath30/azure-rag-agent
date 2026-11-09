import numpy as np

from ragagent.agent import Agent
from ragagent.documents import Chunk
from ragagent.embeddings import TfidfEmbedder
from ragagent.llm import NO_ANSWER, ExtractiveGenerator
from ragagent.vectorstore import InMemoryVectorStore, SearchResult


def test_tfidf_embeddings_are_deterministic_and_normalised() -> None:
    emb = TfidfEmbedder()
    emb.fit(["paid time off policy", "dental plan coverage"])
    a, b = emb.embed(["paid time off"]), emb.embed(["paid time off"])
    assert np.array_equal(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)


def test_in_memory_store_returns_best_match_first() -> None:
    chunks = [Chunk(f"c-{i}", "d", "T", "S", f"text {i}") for i in range(3)]
    store = InMemoryVectorStore()
    store.add(chunks, np.eye(3, dtype=np.float32))
    results = store.search(np.array([0, 1, 0], dtype=np.float32), k=2)
    assert results[0].chunk.chunk_id == "c-1"
    assert results[0].score == 1.0
    assert len(results) == 2


def test_retrieve_finds_expected_doc(agent: Agent) -> None:
    results = agent.pipeline.retrieve("What is the 401k employer match?", k=3)
    assert results[0].chunk.doc_id == "retirement-401k"


def test_ask_returns_answer_with_valid_citations(agent: Agent) -> None:
    answer = agent.pipeline.ask("How much is the home office stipend?")
    assert "$500" in answer.answer
    assert "[1]" in answer.answer
    assert answer.citations[0].doc_id == "remote-work"
    assert [c.number for c in answer.citations] == list(range(1, len(answer.citations) + 1))


def test_out_of_domain_question_gets_no_answer(agent: Agent) -> None:
    answer = agent.pipeline.ask("What is the capital of France?")
    assert answer.answer == NO_ANSWER
    assert answer.citations == []


def test_extractive_generator_with_no_overlap_says_so() -> None:
    result = SearchResult(Chunk("c-0", "d", "T", "S", "Completely unrelated sentence."), 0.9)
    assert ExtractiveGenerator().generate("parental leave weeks", [result]) == NO_ANSWER
