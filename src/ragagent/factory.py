"""Wire up the local or Azure backend from Settings."""

from __future__ import annotations

from typing import Any

from ragagent.agent import Agent, AzureToolPlanner, Planner, RuleBasedPlanner
from ragagent.config import Settings
from ragagent.documents import chunk_documents, load_documents
from ragagent.embeddings import AzureOpenAIEmbedder, Embedder, TfidfEmbedder
from ragagent.llm import AnswerGenerator, AzureChatGenerator, ExtractiveGenerator
from ragagent.rag import RAGPipeline
from ragagent.vectorstore import AzureSearchVectorStore, InMemoryVectorStore, VectorStore


def _azure_openai_client(settings: Settings) -> Any:
    from openai import AzureOpenAI  # optional dependency: pip install ".[azure]"

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )


def build_pipeline(settings: Settings) -> tuple[RAGPipeline, Planner]:
    """Build and index the pipeline; also return the matching agent planner."""
    embedder: Embedder
    store: VectorStore
    generator: AnswerGenerator
    planner: Planner

    if settings.backend == "local":
        embedder, store = TfidfEmbedder(), InMemoryVectorStore()
        generator, planner = ExtractiveGenerator(), RuleBasedPlanner()
    elif settings.backend == "azure":
        missing = settings.missing_azure_settings()
        if missing:
            raise ValueError(f"Azure backend needs these env vars: {', '.join(missing)}")
        client = _azure_openai_client(settings)
        embedder = AzureOpenAIEmbedder(client, settings.azure_openai_embedding_deployment or "")
        store = AzureSearchVectorStore(
            settings.azure_search_endpoint or "",
            settings.azure_search_api_key or "",
            settings.azure_search_index,
        )
        generator = AzureChatGenerator(client, settings.azure_openai_chat_deployment or "")
        planner = AzureToolPlanner(client, settings.azure_openai_chat_deployment or "")
    else:
        raise ValueError(f"Unknown RAG_BACKEND {settings.backend!r}; use 'local' or 'azure'")

    pipeline = RAGPipeline(embedder, store, generator, settings.top_k, settings.min_score)
    docs = load_documents(settings.docs_dir)
    pipeline.index(chunk_documents(docs, settings.chunk_size, settings.chunk_overlap))
    return pipeline, planner


def build_agent(settings: Settings) -> Agent:
    pipeline, planner = build_pipeline(settings)
    return Agent(pipeline, planner)
