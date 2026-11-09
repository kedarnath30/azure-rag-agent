"""FastAPI service: /health, /ask, /agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ragagent import __version__
from ragagent.config import Settings
from ragagent.factory import build_agent


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AgentResponse(BaseModel):
    answer: str
    steps: list[dict[str, Any]]
    citations: list[CitationOut]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.agent = build_agent(settings)  # loads and indexes the docs once
        yield

    app = FastAPI(title="Azure RAG Agent", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "backend": settings.backend, "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        result = request.app.state.agent.pipeline.ask(body.question, body.top_k)
        return AskResponse(
            answer=result.answer,
            citations=[CitationOut(**asdict(c)) for c in result.citations],
        )

    @app.post("/agent", response_model=AgentResponse)
    def agent(body: AgentRequest, request: Request) -> AgentResponse:
        result = request.app.state.agent.run(body.question)
        return AgentResponse(
            answer=result.answer,
            steps=[{"tool": s.call.name, "arguments": s.call.arguments} for s in result.steps],
            citations=[CitationOut(**asdict(c)) for c in result.citations],
        )

    return app


app = create_app()
