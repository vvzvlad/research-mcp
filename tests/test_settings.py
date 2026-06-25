"""Settings: all fields default, so import/construction never fails on ENV."""

from src.settings import Settings


def test_defaults_present():
    s = Settings(_env_file=None)
    assert s.mcp_host == "0.0.0.0"
    assert s.mcp_port == 8000
    assert s.request_timeout == 25.0
    assert s.fallback_min_chars == 400
    assert s.read_pages_concurrency == 5
    assert s.retries == 1


def test_read_pages_max_is_not_a_setting():
    # The per-call url cap is a hard constant (server.READ_PAGES_MAX), not a
    # setting, so the tool's "up to 20" description cannot be made to lie.
    assert "read_pages_max" not in Settings.model_fields


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MCP_PORT", "9001")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FALLBACK_MIN_CHARS", "100")
    s = Settings(_env_file=None)
    assert s.mcp_port == 9001
    assert s.log_level == "DEBUG"
    assert s.fallback_min_chars == 100


def test_no_provider_keys_as_fields():
    # Provider secrets must NOT be declared as Settings fields (read by name in
    # the instance loader instead), keeping the model small.
    fields = set(Settings.model_fields)
    for forbidden in (
        "searxng_url",
        "serper_api_key",
        "exa_api_key",
        "crawl4ai_url",
        "crawl4ai_token",
        "tavily_1_api_key",
        "firecrawl_api_key",
    ):
        assert forbidden not in fields
