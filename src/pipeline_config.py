"""In-code pipeline configuration: which provider instances exist and in what
order each pipeline tries them.

This is intentionally CODE, not YAML/ENV. An ``Instance`` records a *type*, an
instance *name*, and the NAMES of the environment variables that hold its
secrets/URL — never the values themselves. The loader (``src/pipeline.py``)
resolves those names with ``os.getenv`` at startup and enables an instance only
when its required variables are present.

To add a provider:
  1. write ``src/providers/<type>.py`` with an ``@register("<type>")`` class,
  2. import it in ``src/providers/__init__.py``,
  3. add an ``Instance(...)`` line here and reference it in a pipeline below,
  4. document its ENV var in ``.env.example``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Instance:
    """A configured provider instance description.

    ``url_env`` / ``api_key_env`` / ``token_env`` / ``proxy_env`` are ENV
    VARIABLE NAMES, not values. ``optional_api_key`` marks an instance (e.g.
    jina) that may run without its key — it stays enabled even if the key var is
    unset. ``proxy_env`` (optional) names a var holding a SOCKS5/HTTP proxy URL
    to route this instance's outbound requests through; unset → direct egress.
    """

    name: str
    type: str
    url_env: str | None = None
    api_key_env: str | None = None
    token_env: str | None = None
    proxy_env: str | None = None
    optional_api_key: bool = False


# All instances that *could* run. An instance is actually enabled at startup
# only if every required ENV var it names is set (see pipeline.py). Multiple
# instances of one type are allowed (tavily-1 / tavily-2 with different keys).
# External (public-internet) instances carry a `proxy_env` so they can be routed
# through a clean-egress SOCKS5/HTTP proxy (some are IP-blocked by Cloudflare).
# Internal/local instances (searxng, crawl4ai, trafilatura) have NO proxy_env.
INSTANCES: list[Instance] = [
    # --- search ---
    Instance("searxng", "searxng", url_env="SEARXNG_URL"),
    Instance("serper", "serper", api_key_env="SERPER_API_KEY", proxy_env="SERPER_PROXY"),
    Instance("exa", "exa", api_key_env="EXA_API_KEY", proxy_env="EXA_PROXY"),
    # --- read ---
    Instance("trafilatura", "trafilatura"),
    # jina works keyless (lower rate limit); the key is optional.
    Instance(
        "jina", "jina", api_key_env="JINA_API_KEY", proxy_env="JINA_PROXY", optional_api_key=True
    ),
    Instance("crawl4ai", "crawl4ai", url_env="CRAWL4AI_URL", token_env="CRAWL4AI_TOKEN"),
    Instance("tavily-1", "tavily", api_key_env="TAVILY_1_API_KEY", proxy_env="TAVILY_1_PROXY"),
    Instance("tavily-2", "tavily", api_key_env="TAVILY_2_API_KEY", proxy_env="TAVILY_2_PROXY"),
    Instance("firecrawl", "firecrawl", api_key_env="FIRECRAWL_API_KEY", proxy_env="FIRECRAWL_PROXY"),
]

# Order in which enabled instances are tried. Search runs them concurrently and
# merges; read tries them sequentially until one returns enough content.
SEARCH_PIPELINE: list[str] = ["searxng", "serper", "exa"]
READ_PIPELINE: list[str] = [
    "trafilatura",
    "jina",
    "crawl4ai",
    "tavily-1",
    "tavily-2",
    "firecrawl",
]

# Provider TYPES that bill per successful request (external metered APIs). Used
# ONLY for usage accounting in the logs. Self-hosted / free types (searxng,
# trafilatura, crawl4ai) are never counted as paid. jina is metered when an API
# key is configured, so it is classified as paid.
PAID_TYPES: frozenset[str] = frozenset({"serper", "exa", "jina", "tavily", "firecrawl"})
