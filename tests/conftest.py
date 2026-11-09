from __future__ import annotations

from pathlib import Path

import pytest

from ragagent.agent import Agent
from ragagent.config import Settings
from ragagent.factory import build_agent

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(backend="local", docs_dir=ROOT / "data" / "docs")


@pytest.fixture(scope="session")
def agent(settings: Settings) -> Agent:
    return build_agent(settings)
