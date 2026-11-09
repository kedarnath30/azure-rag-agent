"""Vector index backends: in-memory numpy (default) and Azure AI Search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ragagent.documents import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]: ...


class InMemoryVectorStore:
    """Brute-force cosine search. Fine for thousands of chunks; vectors are pre-normalised."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        if self._matrix is None:
            return []
        scores = self._matrix @ vector  # cosine similarity, since rows are unit length
        top = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in top]


class AzureSearchVectorStore:
    """Azure AI Search index with one vector field (HNSW).

    Requires the `azure` extra: pip install ".[azure]".
    """

    def __init__(self, endpoint: str, api_key: str, index_name: str) -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        credential = AzureKeyCredential(api_key)
        self._index_name = index_name
        self._index_client = SearchIndexClient(endpoint, credential)
        self._client: Any = SearchClient(endpoint, index_name, credential)

    def _ensure_index(self, dimensions: int) -> None:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name="default",
            ),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
        )
        index = SearchIndex(name=self._index_name, fields=fields, vector_search=vector_search)
        self._index_client.create_or_update_index(index)

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        self._ensure_index(dimensions=int(vectors.shape[1]))
        documents = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "text": c.text,
                "vector": v.tolist(),
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.merge_or_upload_documents(documents=documents)

    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        from azure.search.documents.models import VectorizedQuery

        query = VectorizedQuery(vector=vector.tolist(), k_nearest_neighbors=k, fields="vector")
        hits = self._client.search(
            search_text=None,
            vector_queries=[query],
            top=k,
            select=["chunk_id", "doc_id", "title", "section", "text"],
        )
        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=h["chunk_id"],
                    doc_id=h["doc_id"],
                    title=h["title"],
                    section=h["section"],
                    text=h["text"],
                ),
                score=float(h["@search.score"]),
            )
            for h in hits
        ]
