"""Instance loading: ENV-name resolution, enable/disable, startup validation."""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from src.config_errors import ConfigError
from src.pipeline import Pipeline, _resolve_instance
from src.pipeline_config import INSTANCES, PAID_TYPES, READ_PIPELINE, SEARCH_PIPELINE
from src.providers.brave import BraveSearch
from src.providers.duckduckgo import DuckDuckGoSearch
from src.providers.jina_search import JinaSearch
from tests.conftest import _clear_provider_env


def _inst(name):
    return next(i for i in INSTANCES if i.name == name)


def test_instances_store_env_names_not_values():
    # Hard rule: the in-code config holds ENV var NAMES, never secret values.
    serper = _inst("serper")
    assert serper.api_key_env == "SERPER_API_KEY"
    crawl4ai = _inst("crawl4ai")
    assert crawl4ai.url_env == "CRAWL4AI_URL"
    assert crawl4ai.token_env == "CRAWL4AI_TOKEN"
    # External instances carry a proxy_env NAME; internal ones do not.
    assert _inst("exa").proxy_env == "EXA_PROXY"
    assert _inst("tavily-1").proxy_env == "TAVILY_1_PROXY"
    assert _inst("searxng").proxy_env is None
    assert _inst("crawl4ai").proxy_env is None
    assert _inst("trafilatura").proxy_env is None
    # No field looks like a real secret/url value.
    for inst in INSTANCES:
        for attr in (inst.url_env, inst.api_key_env, inst.token_env, inst.proxy_env):
            if attr is not None:
                assert attr.isupper()
                assert "://" not in attr


def test_paid_types_classification():
    # Self-hosted / free types (and keyless duckduckgo) are never billed.
    for free in ("searxng", "duckduckgo", "trafilatura", "crawl4ai"):
        assert free not in PAID_TYPES
    # External metered APIs are billed (jina is metered when keyed, jina_search
    # always needs a key; brave's free plan gives 2000 queries a month and then
    # 429s, paid plans bill separately — what is tracked here is metered
    # external calls, not invoices).
    for paid in ("brave", "serper", "exa", "jina", "jina_search", "tavily", "firecrawl"):
        assert paid in PAID_TYPES


def test_search_pipeline_order():
    # Order is load-bearing: dedup keeps the hit from the earlier provider, so
    # this list is the preference order. Proven and free first (searxng, then
    # keyless duckduckgo, then brave, then the tavily/firecrawl search
    # allowances we already pay for as part of their reader keys), then the
    # proven paid workhorse jina-search, then the vendors we hold no key for
    # yet, cheapest first, then serper (dead balance, kept wired) and exa (the
    # most expensive).
    #
    # duckduckgo's slot right behind searxng is the load-bearing one: it is the
    # only always-on search instance, so a deployment that DOES run SearXNG must
    # keep seeing SearXNG's copy of every shared url, not DuckDuckGo's.
    assert SEARCH_PIPELINE == [
        "searxng",
        "duckduckgo",
        "brave",
        "tavily-search",
        "firecrawl-search",
        "jina-search",
        "xmlriver",
        "parallel",
        "octen",
        "linkup",
        "youcom",
        "serper",
        "exa",
    ]


def test_every_pipeline_name_has_an_instance():
    # SEARCH_PIPELINE/READ_PIPELINE reference instances by NAME: a renamed or
    # deleted Instance would otherwise only show up at runtime, as a skipped
    # provider in the log. (Without this, dropping the brave Instance line keeps
    # the suite green while brave silently disappears from production.)
    names = {inst.name for inst in INSTANCES}
    for name in (*SEARCH_PIPELINE, *READ_PIPELINE):
        assert name in names


def test_brave_instance_is_wired():
    # The brave instance must exist and name the exact ENV vars documented in
    # .env.example — this is what actually turns brave on in production.
    brave = _inst("brave")
    assert brave.type == "brave"
    assert brave.api_key_env == "BRAVE_API_KEY"
    assert brave.proxy_env == "BRAVE_PROXY"
    assert not brave.optional_api_key  # no key → no brave
    assert brave.url_env is None
    assert brave.token_env is None


def test_jina_search_instance_is_wired():
    # jina-search deliberately reuses the reader's env vars (one key, one
    # proxy) — and unlike the reader, its key is REQUIRED: s.jina.ai blocks
    # keyless access, so the instance must auto-disable without JINA_API_KEY.
    jina_search = _inst("jina-search")
    assert jina_search.type == "jina_search"
    assert jina_search.api_key_env == "JINA_API_KEY"
    assert jina_search.proxy_env == "JINA_PROXY"
    assert not jina_search.optional_api_key  # no key → no jina-search
    assert jina_search.url_env is None
    assert jina_search.token_env is None


def test_importing_the_providers_package_registers_every_type():
    # Guards step 2 of the add-a-provider checklist: only an import runs the
    # @register decorator, so a type missing from src/providers/__init__.py is
    # simply unknown at startup and its instances are disabled with nothing but a
    # log line to show for it.
    #
    # Deliberately a SUBPROCESS: in this process other test modules have already
    # imported e.g. src.providers.brave directly, which populates REGISTRY as a
    # side effect and would mask exactly the missing import we are looking for.
    code = (
        "import src.providers\n"
        "from src.providers.registry import REGISTRY\n"
        "print(' '.join(sorted(REGISTRY)))\n"
    )
    root = Path(__file__).resolve().parents[1]
    # PYTHONPATH explicitly, not just cwd: `python -c` only prepends cwd to
    # sys.path when safe-path mode is off, so under PYTHONSAFEPATH=1 (or -P) the
    # child would fail with ModuleNotFoundError: src and turn this into a false
    # red. The parent process gets the same thing from `pythonpath = .` in
    # pytest.ini; the child inherits none of that.
    # timeout so a hanging import fails the run instead of wedging CI forever —
    # pytest-timeout is not installed and CI runs a bare pytest.
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    registered = set(proc.stdout.split())
    missing = {inst.type for inst in INSTANCES} - registered
    assert not missing, f"not imported in src/providers/__init__.py: {sorted(missing)}"


def test_build_enables_brave_from_its_key(monkeypatch, settings):
    # End-to-end wiring: BRAVE_API_KEY alone is enough to get a working brave
    # search provider out of Pipeline.build.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    # duckduckgo needs no ENV, so it is always there alongside whatever a key
    # turned on — that is the point of it, see test_build_with_no_env_at_all.
    assert pipe.search_names == ["duckduckgo", "brave"]
    brave = next(p for p in pipe._search if p.name == "brave")
    assert isinstance(brave, BraveSearch)
    assert brave.proxy is None  # no BRAVE_PROXY → direct egress


def test_build_enables_jina_search_from_the_shared_jina_key(monkeypatch, settings):
    # End-to-end wiring: JINA_API_KEY alone yields BOTH a jina-search search
    # provider and the (always-on) jina reader in keyed mode.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("JINA_API_KEY", "k")
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    assert pipe.search_names == ["duckduckgo", "jina-search"]  # duckduckgo is always on
    jina_search = next(p for p in pipe._search if p.name == "jina-search")
    assert isinstance(jina_search, JinaSearch)
    assert "jina" in pipe.read_names  # the same key feeds the reader


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


def test_resolve_proxy_absent_is_none(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EXA_API_KEY", "k")
    config = _resolve_instance(_inst("exa"))
    assert config is not None
    assert config.proxy is None  # no EXA_PROXY → direct egress


def test_resolve_proxy_present_is_used(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("EXA_PROXY", "socks5://proxy.invalid:1080")
    config = _resolve_instance(_inst("exa"))
    assert config is not None
    assert config.proxy == "socks5://proxy.invalid:1080"


def test_missing_proxy_does_not_disable_instance(monkeypatch):
    # A proxy is always optional: an instance with its key set stays enabled
    # even when its proxy var is unset.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("TAVILY_1_API_KEY", "t1")
    config = _resolve_instance(_inst("tavily-1"))
    assert config is not None
    assert config.api_key == "t1"
    assert config.proxy is None


def test_build_threads_proxy_into_provider(monkeypatch, settings):
    # End-to-end: a configured proxy reaches the built provider instance.
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("EXA_PROXY", "socks5://proxy.invalid:1080")
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    exa = next(p for p in pipe._search if p.name == "exa")
    assert exa.proxy == "socks5://proxy.invalid:1080"
    # searxng (internal) has no proxy.
    searxng = next(p for p in pipe._search if p.name == "searxng")
    assert searxng.proxy is None


def test_build_with_no_env_at_all(monkeypatch, settings):
    # This used to assert ConfigError("No search provider enabled"): with no keys
    # the server refused to start, so the "free minimum" was really "first deploy
    # a SearXNG". duckduckgo (keyless, no variable of any kind) closes the search
    # half the way trafilatura closes the read half — a completely empty
    # environment must now BUILD, and search.
    _clear_provider_env(monkeypatch)
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    assert pipe.search_names == ["duckduckgo"]  # the only one that needs nothing
    assert isinstance(pipe._search[0], DuckDuckGoSearch)
    assert pipe.read_names[0] == "trafilatura"  # its read-side counterpart
    # Keyless means unbilled: an empty env must cost nothing.
    assert "duckduckgo" not in pipe._paid


def test_config_error_still_guards_an_empty_search_pipeline(monkeypatch, settings):
    # The ConfigError branch stays: duckduckgo makes it unreachable through ENV,
    # but not through a broken SEARCH_PIPELINE (a renamed instance, a provider
    # whose __init__ raised). Simulated by emptying the pipeline list.
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr("src.pipeline.SEARCH_PIPELINE", [])
    with pytest.raises(ConfigError) as ei:
        Pipeline.build(settings, client=httpx.AsyncClient())
    assert "search provider" in str(ei.value).lower()


def test_duckduckgo_instance_needs_no_variable(monkeypatch):
    # The whole contract of this instance is the absence of config: any env name
    # added here would silently make it switchable off again.
    ddg = _inst("duckduckgo")
    assert ddg.type == "duckduckgo"
    assert ddg.url_env is None
    assert ddg.api_key_env is None
    assert ddg.token_env is None
    assert ddg.proxy_env is None
    _clear_provider_env(monkeypatch)
    assert _resolve_instance(ddg) is not None  # enabled on an empty environment


def test_build_enables_expected_instances(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.test")
    monkeypatch.setenv("TAVILY_2_API_KEY", "t2")
    pipe = Pipeline.build(settings, client=httpx.AsyncClient())
    # searxng among the keyed search instances, with the always-on duckduckgo
    # behind it; trafilatura+jina always-on plus tavily-2 on the read side.
    assert pipe.search_names == ["searxng", "duckduckgo"]
    assert "trafilatura" in pipe.read_names
    assert "jina" in pipe.read_names
    assert "tavily-2" in pipe.read_names
    assert "tavily-1" not in pipe.read_names  # its key is unset
    assert "crawl4ai" not in pipe.read_names
