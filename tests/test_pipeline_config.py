"""Instance loading: ENV-name resolution, enable/disable, startup validation."""

import httpx
import pytest

from src.config_errors import ConfigError
from src.pipeline import Pipeline, _resolve_instance
from src.pipeline_config import INSTANCES


def _inst(name):
    return next(i for i in INSTANCES if i.name == name)


def _clear_provider_env(monkeypatch):
    for var in (
        "SEARXNG_URL",
        "SERPER_API_KEY",
        "EXA_API_KEY",
        "JINA_API_KEY",
        "CRAWL4AI_URL",
        "CRAWL4AI_TOKEN",
        "TAVILY_1_API_KEY",
        "TAVILY_2_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_instances_store_env_names_not_values():
    # Hard rule: the in-code config holds ENV var NAMES, never secret values.
    serper = _inst("serper")
    assert serper.api_key_env == "SERPER_API_KEY"
    crawl4ai = _inst("crawl4ai")
    assert crawl4ai.url_env == "CRAWL4AI_URL"
    assert crawl4ai.token_env == "CRAWL4AI_TOKEN"
    # No field looks like a real secret/url value.
    for inst in INSTANCES:
        for attr in (inst.url_env, inst.api_key_env, inst.token_env):
            if attr is not None:
                assert attr.isupper()
                assert "://" not in attr


def test_resolve_disabled_when_key_missing(monkeypatch):
    _clear_provider_env(monkeypatch)
    assert _resolve_instance(_inst("serper")) is None  # needs SERPER_API_KEY


def test_resolve_enabled_when_key_present(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SERPER_API_KEY", "k")
    config = _resolve_instance(_inst("serper"))
    assert config is not None
    assert config.api_key == "k"


def test_jina_optional_key_enabled_without_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    config = _resolve_instance(_inst("jina"))
    assert config is not None  # jina is keyless-capable → always enabled
    assert config.api_key is None


def test_build_requires_one_search_provider(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    # trafilatura/jina give read providers, but no search provider is enabled.
    with pytest.raises(ConfigError) as ei:
        Pipeline.build(settings, client=httpx.AsyncClient())
    assert "search provider" in str(ei.value).lower()


def test_build_enables_expected_instances(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("TAVILY_2_API_KEY", "t2")
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    # Only searxng among search; trafilatura+jina always-on plus tavily-2.
    assert pipe.search_names == ["searxng"]
    assert "trafilatura" in pipe.read_names
    assert "jina" in pipe.read_names
    assert "tavily-2" in pipe.read_names
    assert "tavily-1" not in pipe.read_names  # its key is unset
    assert "crawl4ai" not in pipe.read_names
