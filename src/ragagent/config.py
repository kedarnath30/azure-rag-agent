"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend: str = "local"  # "local" (offline, default) or "azure"
    docs_dir: Path = Path("data/docs")
    top_k: int = 4
    min_score: float = 0.05  # drop retrieved chunks below this similarity
    chunk_size: int = 120  # words per chunk
    chunk_overlap: int = 20  # words shared by neighbouring chunks

    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-06-01"
    azure_openai_chat_deployment: str | None = None
    azure_openai_embedding_deployment: str | None = None
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "hr-policies"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            backend=env("RAG_BACKEND", "local").lower(),
            docs_dir=Path(env("RAG_DOCS_DIR", "data/docs")),
            top_k=int(env("RAG_TOP_K", "4")),
            min_score=float(env("RAG_MIN_SCORE", "0.05")),
            chunk_size=int(env("RAG_CHUNK_SIZE", "120")),
            chunk_overlap=int(env("RAG_CHUNK_OVERLAP", "20")),
            azure_openai_endpoint=env("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_key=env("AZURE_OPENAI_API_KEY"),
            azure_openai_api_version=env("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_openai_chat_deployment=env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            azure_openai_embedding_deployment=env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            azure_search_endpoint=env("AZURE_SEARCH_ENDPOINT"),
            azure_search_api_key=env("AZURE_SEARCH_API_KEY"),
            azure_search_index=env("AZURE_SEARCH_INDEX", "hr-policies"),
        )

    def missing_azure_settings(self) -> list[str]:
        """Names of the env vars the Azure backend needs but that are not set."""
        required = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": self.azure_openai_chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.azure_openai_embedding_deployment,
            "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
            "AZURE_SEARCH_API_KEY": self.azure_search_api_key,
        }
        return [name for name, value in required.items() if not value]
