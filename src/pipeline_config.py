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
# No `proxy_env` for the instances that do not need clean egress: the internal
# ones (searxng/crawl4ai/trafilatura — they still name their own url/token vars)
# and duckduckgo, which is external but keyless.
INSTANCES: list[Instance] = [
    # --- search ---
    Instance("searxng", "searxng", url_env="SEARXNG_URL"),
    # No key, no url, no variable of any kind: the only search instance that is
    # ALWAYS enabled. It is the search half of the zero-config floor (the
    # trafilatura of searching), which is what keeps `Pipeline.build` working on
    # a completely empty environment — before it, "free minimum" meant "first
    # deploy a SearXNG". It scrapes the no-JS SERP, so it is free but fragile:
    # it sits behind searxng in the pipeline, never in front of it.
    Instance("duckduckgo", "duckduckgo"),
    # Brave reaches its API fine from this host directly (verified 2026-08-09);
    # the proxy var exists only for symmetry with the other external instances.
    Instance("brave", "brave", api_key_env="BRAVE_API_KEY", proxy_env="BRAVE_PROXY"),
    # Reuses the same key (and the same token pool) as the jina reader below.
    # Unlike the reader, s.jina.ai refuses keyless access, so the key is
    # REQUIRED here — no optional_api_key — and the instance simply
    # auto-disables when JINA_API_KEY is unset.
    Instance("jina-search", "jina_search", api_key_env="JINA_API_KEY", proxy_env="JINA_PROXY"),
    # Tavily and Firecrawl sell search and extract off ONE key, so these two
    # need no new secret and light up the moment the reader key is present.
    #
    # They are NOT free, though: the key has ONE monthly pool, shared by both
    # products. Measured on our own Tavily key 2026-09-18: plan_usage 12 =
    # search 4 + extract 8, against plan_limit 1000. Search runs on every
    # web_search while the tavily/firecrawl READERS sit 4th and 6th in the read
    # chain and win ~1% of reads (19-27 calls a month), so search will be what
    # empties the pool — and when it does, those readers start failing too and
    # drop out of the chain. Firecrawl says so with a 402 (seen in our own logs);
    # what status Tavily uses is NOT verified — its docs do not publish one, so
    # do not grep for a specific code there, and note that _CREDIT_MARKERS may
    # or may not match its wording. That trade is deliberate: the readers we
    # lose are worth far less than the search we gain, and the chain has five
    # other readers. But it is a trade, not a free lunch, and it cannot be
    # turned off per-product — pulling the key disables the reader as well.
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
# Order: proven and free first (searxng self-hosted, then keyless duckduckgo,
# then brave's 2000/month), then tavily-search and firecrawl-search — no new
# secret, but NOT free: they spend the same monthly pool as their own readers,
# see the instance comment above. Then the proven paid workhorse jina-search at
# ~$0.0005/query, then the vendors we have no key for yet, cheapest first
# (xmlriver ~$0.0003 on the Yandex index, parallel and octen at $1/1k, linkup
# and youcom at $5/1k), then serper, then exa at $7/1k.
#
# duckduckgo sits DIRECTLY AFTER searxng, and that position is load-bearing:
# it needs no key, so it is on in every deployment, but it is a scrape of a
# public SERP rather than an API. Behind searxng, dedup (which prefers the
# earlier source) keeps SearXNG's copy of every shared url, and duckduckgo adds
# only what nobody ahead of it returned. Note what this does NOT promise: with
# the reranker on (the default once JINA_API_KEY is set) the merged list is
# reordered by relevance across ALL sources before the trim, so the final order
# — and which hits survive num_results — can change.
SEARCH_PIPELINE: list[str] = [
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
# trafilatura, crawl4ai) and keyless ones (duckduckgo) are never counted as
# paid. jina is metered when an API key is configured, so it is classified as
# paid. brave is metered too: the free plan grants a 2000-queries-per-month quota
# and answers 429 once it is spent (there is no overage on it); the paid plans
# are billed separately. It is listed here because the accounting tracks metered
# external calls, not invoices.
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
