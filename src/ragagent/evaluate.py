"""Offline evaluation: retrieval hit@k / MRR and citation metrics on a golden set."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ragagent.rag import RAGPipeline

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class GoldenItem:
    question: str
    expected_doc: str
    expected_keywords: list[str]


def load_golden(path: Path) -> list[GoldenItem]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [GoldenItem(**json.loads(line)) for line in lines if line.strip()]


def evaluate(
    pipeline: RAGPipeline, items: list[GoldenItem], ks: tuple[int, ...] = (1, 3, 5)
) -> dict:
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    reciprocal_ranks = 0.0
    has_citation = cites_expected = has_keywords = 0
    failures: list[dict] = []

    for item in items:
        retrieved = [r.chunk.doc_id for r in pipeline.retrieve(item.question, k=max_k)]
        rank = retrieved.index(item.expected_doc) + 1 if item.expected_doc in retrieved else None
        for k in ks:
            hits[k] += int(rank is not None and rank <= k)
        reciprocal_ranks += 1 / rank if rank else 0.0

        answer = pipeline.ask(item.question)
        numbers = {int(n) for n in CITATION.findall(answer.answer)}
        valid = [c for c in answer.citations if c.number in numbers]
        has_citation += int(bool(valid))
        cites_expected += int(any(c.doc_id == item.expected_doc for c in valid))
        keywords_ok = all(kw.lower() in answer.answer.lower() for kw in item.expected_keywords)
        has_keywords += int(keywords_ok)
        if rank != 1 or not keywords_ok:
            failures.append(
                {
                    "question": item.question,
                    "expected_doc": item.expected_doc,
                    "rank": rank,
                    "answer": answer.answer,
                }
            )

    n = len(items)
    metrics = {f"hit@{k}": round(hits[k] / n, 3) for k in ks}
    metrics.update(
        {
            "mrr": round(reciprocal_ranks / n, 3),
            "answer_has_citation": round(has_citation / n, 3),
            "answer_cites_expected_doc": round(cites_expected / n, 3),
            "answer_contains_keywords": round(has_keywords / n, 3),
        }
    )
    return {"n_questions": n, "metrics": metrics, "failures": failures}
