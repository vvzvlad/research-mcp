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

from src.quota import Quota


@dataclass(frozen=True, slots=True)
class Instance:
    """A configured provider instance description.

    ``url_env`` / ``api_key_env`` / ``token_env`` / ``proxy_env`` are ENV
    VARIABLE NAMES, not values. ``optional_api_key`` marks an instance (e.g.
    jina) that may run without its key — it stays enabled even if the key var is
    unset. ``proxy_env`` (optional) names a var holding a SOCKS5/HTTP proxy URL
    to route this instance's outbound requests through; unset → direct egress.
    ``options_env`` pairs an option key with the ENV VARIABLE NAME holding its
    value; always optional — each var that is set lands in
    ``ProviderConfig.options`` under its key, an unset one is simply left out.
    ``quota`` (optional) caps the instance's calls per period in the persistent
    quota store (``src/quota.py``), keyed by instance name; None → unlimited.
    ``disabled_by_env`` names a var which, set to a non-empty value, switches the
    instance off (empty counts as unset, like every var here) — for an instance
    that steps aside when another one covers the same upstream.
    """

    name: str
    type: str
    url_env: str | None = None
    api_key_env: str | None = None
    token_env: str | None = None
    proxy_env: str | None = None
    optional_api_key: bool = False
    options_env: tuple[tuple[str, str], ...] = ()
    quota: Quota | None = None
    disabled_by_env: str | None = None


# All instances that *could* run. An instance is actually enabled at startup
# only if every required ENV var it names is set and its `disabled_by_env` var,
# if any, is not (see pipeline.py). Multiple
# instances of one type are allowed (tavily-1 / tavily-2 with different keys).
# External (public-internet) instances carry a `proxy_env` so they can be routed
# through a clean-egress SOCKS5/HTTP proxy (some are IP-blocked by Cloudflare).
# No `proxy_env` for the instances that do not need clean egress: the internal
# ones (searxng/crawl4ai/trafilatura — they still name their own url/token vars)
# and duckduckgo, which is external but keyless.
INSTANCES: list[Instance] = [
    # --- search ---
    Instance("searxng", "searxng", url_env="SEARXNG_URL"),
    # No key, no url: the search half of the zero-config floor (the trafilatura
    # of searching), which is what keeps `Pipeline.build` working on a
    # completely empty environment. It scrapes the no-JS SERP, so it is free but
    # fragile.
    #
    # It steps aside when SEARXNG_URL is set: our SearXNG queries the very same
    # html.duckduckgo.com endpoint from the same egress IP (DuckDuckGo is the only
    # engine it still has), and DuckDuckGo already answers it with CAPTCHAs many
    # times a day — a second concurrent query per search only adds to that.
    Instance("duckduckgo", "duckduckgo", disabled_by_env="SEARXNG_URL"),
    # Brave reaches its API directly from this host; the proxy var exists only
    # for symmetry with the other external instances.
    # The free plan grants 2000 queries a month (see src/providers/brave.py).
    Instance(
        "brave",
        "brave",
        api_key_env="BRAVE_API_KEY",
        proxy_env="BRAVE_PROXY",
        quota=Quota(2000, "monthly"),
    ),
    # Reuses the same key (and the same token pool) as the jina reader below.
    # Unlike the reader, s.jina.ai refuses keyless access, so the key is
    # REQUIRED here — no optional_api_key — and the instance simply
    # auto-disables when JINA_API_KEY is unset.
    Instance("jina-search", "jina_search", api_key_env="JINA_API_KEY", proxy_env="JINA_PROXY"),
    # Tavily and Firecrawl sell search and extract off ONE key, so these two
    # need no new secret and light up the moment the reader key is present.
    #
    # They are NOT free: the key has ONE monthly pool shared by both products.
    # Search runs on every web_search while these readers win few reads, so
    # search is what empties the pool, and then the readers fail too. Firecrawl
    # signals it with a 402; Tavily's status is undocumented, and _CREDIT_MARKERS
    # may not match its wording. It cannot be turned off per product — pulling
    # the key disables the reader as well.
    #
    # The pool is 1000 calls a month; these readers sit 4th and 6th in the read
    # chain and win ~1% of reads (19-27 a month). Losing them when search drains
    # the pool is a deliberate trade: the chain has five other readers.
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
    # Scholarly indexes: papers mixed into every web_search. Free with a key;
    # keyless is no option (OpenAlex allows ~100 searches a day without one,
    # Semantic Scholar answers 429 from the first request), so no key, no instance.
    Instance(
        "openalex", "openalex_search", api_key_env="OPENALEX_API_KEY", proxy_env="OPENALEX_PROXY"
    ),
    Instance(
        "semantic-scholar",
        "semantic_scholar_search",
        api_key_env="SEMANTIC_SCHOLAR_API_KEY",
        proxy_env="SEMANTIC_SCHOLAR_PROXY",
    ),
    # --- read ---
    # The url-specific readers: each serves only the urls it accepts, ahead of
    # the probe. Their hosts are fixed, so they fetch with the plain client bound
    # to their proxy, not the SSRF-guarded one — except scholarly, which fetches
    # the third-party urls Unpaywall names and so gets the guarded client.
    # YouTube transcripts need no key. Some hosts (prod among them) cannot reach
    # youtube.com directly, or are flagged by it as a bot — hence the proxy.
    Instance("youtube", "youtube", proxy_env="YOUTUBE_PROXY"),
    # Paid fallback for YouTube transcripts when YouTube bot-walls our IP; native captions only.
    Instance("supadata", "supadata", api_key_env="SUPADATA_API_KEY", proxy_env="SUPADATA_PROXY"),
    # Instagram post transcripts: Groq Whisper transcribes the audio, so the key
    # is Groq's and no key means no instance. Two upstreams, two proxies:
    # INSTAGRAM_PROXY for instagram.com where Instagram is blocked, GROQ_PROXY
    # for api.groq.com where Groq answers "Forbidden" for the egress country.
    Instance(
        "instagram",
        "instagram",
        api_key_env="GROQ_API_KEY",
        proxy_env="INSTAGRAM_PROXY",
        options_env=(("groq_proxy", "GROQ_PROXY"),),
    ),
    # An Instagram profile's posts need no key; same instagram.com proxy.
    Instance("instagram-profile", "instagram_profile", proxy_env="INSTAGRAM_PROXY"),
    # Free OA full-text resolution for DOI/arXiv links via Unpaywall + Europe PMC + arXiv.
    Instance(
        "scholarly",
        "scholarly",
        proxy_env="SCHOLARLY_PROXY",
        options_env=(("unpaywall_email", "UNPAYWALL_EMAIL"),),
    ),
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
# Order: free first (searxng self-hosted, keyless duckduckgo, brave's
# 2000/month), then tavily-search and firecrawl-search (they spend their
# readers' monthly pool, see above), then jina-search (~$0.0005/query), then
# the rest cheapest first (xmlriver ~$0.0003 on the Yandex index, parallel and
# octen at $1/1k, linkup and youcom at $5/1k), then serper, then exa at $7/1k.
# The free scholarly indexes (openalex, semantic-scholar) close the list: they
# add papers, and a paper url a web engine also returned keeps the web copy.
#
# searxng and duckduckgo never run together (duckduckgo steps aside when
# SEARXNG_URL is set, see its instance), so whichever of the two is enabled
# heads the list.
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
    "openalex",
    "semantic-scholar",
]
# Read is sequential and stops at the first sufficient answer, so here the order
# IS a cost gate: brightdata sits last because it is the only one that bills for
# pages the cheap providers already handle. The url-specific readers come first:
# they run before the probe, on the urls they accept only.
READ_PIPELINE: list[str] = [
    "youtube",
    "supadata",
    "instagram",
    "instagram-profile",
    "scholarly",
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
# external calls, not invoices. openalex_search likewise: every search is charged
# against the key's free daily budget. semantic_scholar_search is only rate-limited.
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
        "instagram",  # Groq bills every transcription
        "supadata",  # 1 credit per native transcript; Supadata also bills a 206 this does not count
        "openalex_search",
    }
)
