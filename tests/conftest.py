"""Shared pytest fixtures.

Settings now has only defaulted fields, so importing it needs no ENV. Provider
instances are selected by the ENV var NAMES in pipeline_config; tests set those
vars explicitly (usually via monkeypatch) to choose which instances are enabled.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from loguru import logger

from src.providers.base import ProviderConfig
from src.providers.duckduckgo import DDG_ENDPOINT
from src.settings import Settings


# Every provider ENV var the instance loader looks at. Tests clear them all
# before setting the few they want, so a variable that happens to be exported in
# the developer's shell cannot silently enable an extra instance. Lives here (not
# in a test module) because several test modules need it — keep it in sync with
# ``INSTANCES`` in src/pipeline_config.py.
_PROVIDER_ENV_VARS = (
    "SEARXNG_URL",
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "EXA_API_KEY",
    "JINA_API_KEY",
    "CRAWL4AI_URL",
    "CRAWL4AI_TOKEN",
    "TAVILY_1_API_KEY",
    "TAVILY_2_API_KEY",
    "FIRECRAWL_API_KEY",
    "BRAVE_PROXY",
    "SERPER_PROXY",
    "EXA_PROXY",
    "JINA_PROXY",
    "TAVILY_1_PROXY",
    "TAVILY_2_PROXY",
    "FIRECRAWL_PROXY",
    "XMLRIVER_API_KEY",
    "XMLRIVER_USER_ID",
    "XMLRIVER_PROXY",
    "PARALLEL_API_KEY",
    "PARALLEL_PROXY",
    "OCTEN_API_KEY",
    "OCTEN_PROXY",
    "LINKUP_API_KEY",
    "LINKUP_PROXY",
    "YOUCOM_API_KEY",
    "YOUCOM_PROXY",
    "BRIGHTDATA_API_KEY",
    "BRIGHTDATA_ZONE",
    "BRIGHTDATA_PROXY",
)


def _clear_provider_env(monkeypatch) -> None:
    """Unset every provider ENV var so a test controls the enabled instances."""
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _mock_duckduckgo_rate_limited() -> None:
    """Route the always-on duckduckgo instance to its own soft-block answer.

    duckduckgo names NO env var (that is the whole point of it), so
    ``_clear_provider_env`` cannot switch it off: every ``Pipeline.build``
    enables it and every pipeline-level search fires a POST at it. A test about
    OTHER providers calls this so the instance fails deterministically — 202 +
    "Ratelimit" is DuckDuckGo's own rate-limit block, which the provider turns
    into a ``ProviderError``, so it drops out of the merge and out of the
    ``providers=[...]`` log field exactly as an unconfigured instance would.

    Without it the request is merely unmatched: respx raises "not mocked", which
    happens to keep those tests green but would let a run with
    ``assert_all_mocked`` turned off hit the real DuckDuckGo.
    """
    respx.post(DDG_ENDPOINT).mock(return_value=httpx.Response(202, text="Ratelimit"))


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
