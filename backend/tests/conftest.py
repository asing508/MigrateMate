"""Shared test configuration.

The whole suite runs fully offline: no Gemini key, no Docker services, no
torch import. Env vars are set *before* the app is imported so settings pick
them up, and an autouse fixture forces the migration agent into its
pattern-based (no-network) fallback.
"""

import os

# Must be set before app.core.config is imported anywhere.
os.environ["GEMINI_API_KEY"] = ""          # force pattern-based migration
os.environ["ENABLE_VECTOR_SERVICES"] = "false"
os.environ["ENABLE_DATABASE"] = "false"

import pytest


@pytest.fixture(autouse=True)
def _offline_agent():
    """Guarantee the migration agent never calls out to Gemini during tests."""
    from app.agents.migration_agent import get_migration_agent
    agent = get_migration_agent()
    agent._model = None
    yield
