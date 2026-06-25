"""Shared pytest fixtures.

Settings now has only defaulted fields, so importing it needs no ENV. Provider
instances are selected by the ENV var NAMES in pipeline_config; tests set those
vars explicitly (usually via monkeypatch) to choose which instances are enabled.
"""

from __future__ import annotations

import pytest
from loguru import logger

from src.providers.base import ProviderConfig
from src.settings import Settings


@pytest.fixture
def capture_logs():
    """Capture loguru messages into a list of formatted strings for the test.

    pytest's ``caplog`` does not see loguru records (separate logging stack), so
    we attach a temporary sink and remove it afterwards.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@pytest.fixture
def settings() -> Settings:
    """Settings with small, test-friendly knobs (no env file)."""
    return Settings(
        _env_file=None,
        request_timeout=5.0,
        fallback_min_chars=400,
        read_pages_concurrency=5,
        retries=1,
    )


@pytest.fixture
def make_config():
    """Factory for a ProviderConfig with a given name + resolved secrets/url."""

    def _make(name: str, **kwargs) -> ProviderConfig:
        return ProviderConfig(name=name, request_timeout=5.0, retries=1, **kwargs)

    return _make
