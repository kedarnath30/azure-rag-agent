"""Embedding backends: offline TF-IDF (default) and Azure OpenAI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: Sequence[str]) -> None:
        """Learn anything the embedder needs from the corpus (no-op for API models)."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return one L2-normalised vector per text, shape (len(texts), dim)."""


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class TfidfEmbedder:
    """Deterministic local embedding: TF-IDF over unigrams and bigrams.

    Not semantic like a neural model, but it needs no network or API key,
    which is what lets the tests and CI run fully offline.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed()")
        matrix = self._vectorizer.transform(texts).toarray().astype(np.float32)
        return _normalize(matrix)


class AzureOpenAIEmbedder:
    """Embeddings from an Azure OpenAI deployment (e.g. text-embedding-3-small)."""

    def __init__(self, client: Any, deployment: str) -> None:
        self._client = client  # an openai.AzureOpenAI instance
        self._deployment = deployment

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._deployment, input=list(texts))
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return _normalize(vectors)
