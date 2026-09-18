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
    # Brave reaches its API fine from this host directly (verified 2026-08-09);
    # the proxy var exists only for symmetry with the other external instances.
    Instance("brave", "brave", api_key_env="BRAVE_API_KEY", proxy_env="BRAVE_PROXY"),
    # Reuses the same key (and the same token pool) as the jina reader below.
    # Unlike the reader, s.jina.ai refuses keyless access, so the key is
    # REQUIRED here — no optional_api_key — and the instance simply
    # auto-disables when JINA_API_KEY is unset.
    Instance("jina-search", "jina_search", api_key_env="JINA_API_KEY", proxy_env="JINA_PROXY"),
    # Tavily and Firecrawl sell search and extract off ONE key: the read
    # instances below already carry these vars, and the search half of both free
    # monthly allowances (1000 and ~500 calls) was going unused. So these two
    # instances cost nothing new and need no extra secret — they light up the
    # moment the reader key is present.
    Instance(
        "tavily-search", "tavily_search", api_key_env="TAVILY_1_API_KEY", proxy_env="TAVILY_1_PROXY"
    ),
    Instance(
        "firecrawl-search",
        "firecrawl_search",
        api_key_env="FIRECRAWL_API_KEY",
        proxy_env="FIRECRAWL_PROXY",
    ),
    # Everything from here down has NO key yet: each stays disabled (one log
    # line at startup) until its vars are set. XMLRiver needs two — the numeric
    # account id travels as `user` and the key as `key`, both in the query
    # string — and defaults to the Yandex index, which is the reason to have it.
    Instance(
        "xmlriver",
        "xmlriver_search",
        api_key_env="XMLRIVER_API_KEY",
        token_env="XMLRIVER_USER_ID",
        proxy_env="XMLRIVER_PROXY",
    ),
    Instance(
        "parallel", "parallel_search", api_key_env="PARALLEL_API_KEY", proxy_env="PARALLEL_PROXY"
    ),
    Instance("octen", "octen_search", api_key_env="OCTEN_API_KEY", proxy_env="OCTEN_PROXY"),
    Instance("linkup", "linkup_search", api_key_env="LINKUP_API_KEY", proxy_env="LINKUP_PROXY"),
    Instance("youcom", "youcom_search", api_key_env="YOUCOM_API_KEY", proxy_env="YOUCOM_PROXY"),
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
    # Last resort of the read chain: an anti-bot unlocker for the pages every
    # other provider bounces off (marketplaces, Cloudflare interstitials). Two
    # vars because the zone is an account-side setting, not a secret.
    Instance(
        "brightdata",
        "brightdata",
        api_key_env="BRIGHTDATA_API_KEY",
        token_env="BRIGHTDATA_ZONE",
        proxy_env="BRIGHTDATA_PROXY",
    ),
]

# Order in which enabled instances are tried. Search runs them concurrently and
# merges; read tries them sequentially until one returns enough content.
#
# This list is a DEDUP PREFERENCE, not a cost gate: search fires every enabled
# instance on every query and merges, so the order decides only which copy of a
# duplicate url survives (the earlier one) and which instance gets named as its
# source. Cost scales with how many instances are ENABLED, not with position —
# so enabling all of them means paying all of them on every single query.
#
# Order: proven and free first (searxng self-hosted, brave's 2000/month, then
# tavily and firecrawl, whose search allowance we already own), then the proven
# paid workhorse jina-search at ~$0.0005/query, then the vendors we have no key
# for yet, cheapest first (xmlriver ~$0.0003 on the Yandex index, parallel and
# octen at $1/1k, linkup and youcom at $5/1k), then serper, then exa at $7/1k.
SEARCH_PIPELINE: list[str] = [
    "searxng",
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
# Read is sequential and stops at the first sufficient answer, so here the order
# IS a cost gate: brightdata sits last because it is the only one that bills for
# pages the cheap providers already handle.
READ_PIPELINE: list[str] = [
    "trafilatura",
    "jina",
    "crawl4ai",
    "tavily-1",
    "tavily-2",
    "firecrawl",
    "brightdata",
]

# Provider TYPES that bill per successful request (external metered APIs). Used
# ONLY for usage accounting in the logs. Self-hosted / free types (searxng,
# trafilatura, crawl4ai) are never counted as paid. jina is metered when an API
# key is configured, so it is classified as paid. brave is metered too: the free
# plan grants a 2000-queries-per-month quota and answers 429 once it is spent
# (there is no overage on it); the paid plans are billed separately. It is listed
# here because the accounting tracks metered external calls, not invoices.
PAID_TYPES: frozenset[str] = frozenset(
    {
        "brave",
        "serper",
        "exa",
        "jina",
        "jina_search",
        "tavily",
        "firecrawl",
        "tavily_search",
        "firecrawl_search",
        "xmlriver_search",
        "parallel_search",
        "octen_search",
        "linkup_search",
        "youcom_search",
        "brightdata",
    }
)
