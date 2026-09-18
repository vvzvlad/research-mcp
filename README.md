# research-mcp

A stateless **MCP facade** that hides a pyramid of search/read providers behind a
single streamable-http MCP endpoint and exposes just **3 clean tools** with good
Russian help texts. An LLM gets a simple "search → read" toolset; behind it,
several providers are tried, merged, and failed over automatically.

The app does **no authentication** — it is published through Traefik + basicAuth
on the host. It holds no application state: the only thing persisted is a log
file under `data/` (kept on a volume).

## Works with zero keys

`make run` on an empty `.env` is enough — **no key, no SearXNG, nothing to deploy
first.** Two instances need no configuration at all and are therefore always
enabled: **`duckduckgo`** (search — the no-JS `html.duckduckgo.com` SERP, one
request per query, no token handshake) and **`trafilatura`** (read — local
HTML→Markdown extraction). Together they are the floor: search and read both
work out of the box.

Everything else is an upgrade on top of that floor. A self-hosted **SearXNG**
(`SEARXNG_URL`) is *optional*; when it is configured it sits **ahead** of
DuckDuckGo in the pipeline, so SearXNG keeps its copy of every url both of them
return and DuckDuckGo only adds what nobody ahead of it had. (With the reranker
on — the default once `JINA_API_KEY` is set — the merged list is reordered by
relevance across all sources before the trim, so the final ordering can still
change.) Every paid vendor lights up the moment its key appears.

The floor is deliberately modest: DuckDuckGo is a scraped SERP, not an API, so it
paces itself to one query per 45s (skipping, never waiting, when the slot is
taken) and reports a block or a captcha as a failure rather than as an empty
result set. That pace is the one measured for this upstream: DuckDuckGo blocks
by IP for 7-8 minutes after a burst, and a block taken here would also silence
a SearXNG that reaches DuckDuckGo over the same address.

## Tools

| Tool | What it does |
|------|--------------|
| `web_search(query, num_results=8, page=1, language=None)` | Search across all enabled providers, merge + dedup → ranked list (title, URL, snippet). Search only. |
| `read_page(url)` | One page or PDF → clean Markdown. Auto-detects type, walks the read pipeline (light → heavy) until one succeeds. |
| `read_pages(urls)` | Up to 20 urls concurrently → list of `{url, ok, markdown\|error}`. |

## Architecture: types + instances

Providers are **plugins**. We separate:

- **type** — an implementation class (e.g. the `searxng` search provider), one
  per module in `src/providers/`, registered with `@register("type")`.
- **instance** — a configured copy of a type with its secrets/URL resolved from
  **named environment variables** (multiple instances of one type are allowed,
  e.g. `tavily-1` / `tavily-2` with different keys).

Which instances exist and the order each pipeline tries them is configured **in
code** (`src/pipeline_config.py`); keys/URLs come **from ENV by variable name**.

- **Search pipeline** (`searxng → duckduckgo → brave → tavily-search →
  firecrawl-search → jina-search → xmlriver → parallel → octen → linkup → youcom
  → serper → exa`):
  enabled instances run concurrently; results are merged and deduplicated by
  normalized URL (earlier pipeline position wins). Position is therefore a
  **dedup preference, not a cost gate** — every enabled instance is called on
  every query, so cost scales with how many keys are set. When `JINA_API_KEY` is set (and
  `SEARCH_RERANK_ENABLED` is not turned off), the full merged list is then
  reranked by `jina-reranker-v3.5` so the trim to `num_results` keeps the most
  relevant hits instead of a blind pipeline-order prefix; any rerank failure
  falls back to the merge order.
  `searxng`, `duckduckgo` and `brave` additionally throttle themselves locally
  (one query per 45s, 45s and 1.1s respectively, each matching a measured
  upstream limit — DuckDuckGo shares SearXNG's, since both reach the same engine
  from the same address); when the slot is taken they **skip** the current
  search instead of waiting for it. `duckduckgo` is the only search instance
  that needs no configuration, which is why it sits directly behind
  `searxng`: it is on everywhere, but a deployment that runs SearXNG must
  keep SearXNG's copy of every shared url.
- **Read pipeline** (`trafilatura → jina → crawl4ai → tavily-1 → tavily-2 →
  firecrawl → brightdata`): here the order IS a cost gate — it stops at the
  first sufficient answer, and `brightdata` (the anti-bot unlocker) sits last so
  it only ever sees pages everything cheaper already bounced off. A single probe
  GET classifies the url. PDFs (Content-Type /
  `.pdf` / `%PDF` magic) are extracted with pypdf; for HTML, that same body is
  handed to `trafilatura` so the hot path never GETs twice, then the remaining
  instances are tried in order and the first to return content
  `>= FALLBACK_MIN_CHARS` wins.

Cross-cutting: one transient retry (5xx / transport errors) with a short backoff;
**402 (out of credits) / 429 (rate limited) are treated as a provider failure →
next instance** (this is what makes `tavily-1 → tavily-2` fail over). Vendors
that report an empty balance with some other 4xx — serper answers `400 {"message":
"Not enough credits"}`, octen `403 "Insufficient balance"` — are recognised by
the body and logged as `out of credits` too, so an unpaid account never reads as
a broken API.

An instance is **enabled** only if its required env var(s) are set; otherwise it
is skipped with a log line. `duckduckgo` and `trafilatura` need no config (always
on); `jina` works keyless (its key is optional). At startup the server requires at
least one search and one read instance — a condition those two always satisfy, so
the check now only catches a broken `pipeline_config.py`.

## Adding a provider

1. Write `src/providers/<type>.py` with a class decorated `@register("<type>")`
   implementing `SearchProvider.search(...)` or `ReadProvider.read(...)`.
2. Import the module in `src/providers/__init__.py` (so the decorator runs).
3. Add an `Instance("name", "<type>", api_key_env="YOUR_ENV_NAME")` line in
   `src/pipeline_config.py` and reference its `name` in `SEARCH_PIPELINE` /
   `READ_PIPELINE`. **Use the ENV var NAME, never a value.**
4. Document the env var in `.env.example`.

## Quick start

```bash
make install                # create .venv + install dev/test deps
cp .env.example .env        # fill in the keys you have  (shortcut: make env)
make test                   # run tests
make run                    # run the server (streamable-http on MCP_HOST:MCP_PORT, endpoint /mcp)
```

## Configuration

All config comes from ENV / `.env` (see `.env.example`). Provider secrets/URLs
are read by **name** in the instance loader, not declared as Settings fields. The
non-secret knobs (all defaulted): `MCP_HOST`, `MCP_PORT`, `LOG_LEVEL`,
`LOG_FILE`, `LOG_ROTATION`, `LOG_RETENTION`, `REQUEST_TIMEOUT`,
`FALLBACK_MIN_CHARS`, `READ_PAGES_CONCURRENCY`, `RETRIES`,
`SEARCH_RERANK_ENABLED`, `JINA_TOKEN_BUDGET`, `ALLOW_PRIVATE_NETWORK` (escape
hatch for the SSRF guard: `true` lets `read_page` fetch private/loopback
addresses, which are blocked by default). The `read_pages`
per-call url cap is a fixed `20` (hard constant, matching the tool description) —
not configurable.

Provider env vars — `duckduckgo` and `trafilatura` take none and are always on;
everything below is optional on top of them: `SEARXNG_URL`, `BRAVE_API_KEY`,
`SERPER_API_KEY`, `EXA_API_KEY`, `JINA_API_KEY`
(one key enables the `jina` reader in keyed mode, the `jina-search` provider and
the search reranker; the reader alone also works keyless), `CRAWL4AI_URL` +
`CRAWL4AI_TOKEN`, `TAVILY_1_API_KEY`, `TAVILY_2_API_KEY`, `FIRECRAWL_API_KEY`.
The Tavily and Firecrawl keys each enable **two** instances — the reader and the
search provider — because both vendors sell search and extract off one key.
Keyless until registered: `XMLRIVER_USER_ID` + `XMLRIVER_API_KEY` (Yandex SERP),
`PARALLEL_API_KEY`, `OCTEN_API_KEY`, `LINKUP_API_KEY`, `YOUCOM_API_KEY`, and
`BRIGHTDATA_API_KEY` + `BRIGHTDATA_ZONE`.

## Proxy

Any external instance can be routed through its own **SOCKS5/HTTP proxy** by
setting `<INSTANCE>_PROXY` — useful for clean egress past IP-based blocks (e.g.
Cloudflare in front of Exa). Supported per instance: `EXA_PROXY`, `BRAVE_PROXY`, `SERPER_PROXY`,
`JINA_PROXY`, `TAVILY_1_PROXY`, `TAVILY_2_PROXY`, `FIRECRAWL_PROXY`,
`XMLRIVER_PROXY`, `PARALLEL_PROXY`, `OCTEN_PROXY`, `LINKUP_PROXY`,
`YOUCOM_PROXY`, `BRIGHTDATA_PROXY`. The instances that do not need clean egress
have no proxy: the internal `searxng` / `crawl4ai` / `trafilatura` (which still
take their own url/token vars) and the keyless `duckduckgo`.

The value is passed straight to httpx; `socks5://host:port` does **proxy-side
DNS** (the target hostname is resolved by the proxy, like `curl
--socks5-hostname`), and `socks5h://` / `http://host:port` are also accepted.
Unset → that instance goes direct. The pipeline keeps one pooled httpx client
per distinct proxy URL (and one direct client), selected per instance, so
proxied and direct providers run side by side. Needs the `socks` extra
(`httpx[socks]`, already pinned).

## Logging

Besides stderr (captured by Docker's rotation-capped json-file driver), the
server writes a **persistent log file** to `data/research-mcp.log` (default;
`LOG_ROTATION=20 MB`, `LOG_RETENTION=14 days`). It lives on the `data/` volume,
so it survives container restarts and image updates. The file carries one
**per-request line** per tool call — search (`query`, which provider instances
actually ran, result count, latency) and read (`url`, the winning provider/tier
or `pdf`, `ok`, latency), plus a `read_pages count=N ok=K` summary — making it
useful for analyzing how requests distribute across provider tiers. No request
bodies or secrets are logged, only urls/queries, provider names, counts, timings.

## Deployment

Gitea Actions builds the image and pushes it to the Gitea registry
`gitea.vvzvlad.xyz/projects/research-mcp` (`test` → `build`, tags `latest` +
`sha`). On prod we pull the prebuilt image via `docker-compose.yml` (behind
Traefik + basicAuth, watchtower auto-updates `latest`; the `data/` volume keeps
the log file across updates) — we never build on prod.

## Layout

| Path | Purpose |
|------|---------|
| `src/providers/base.py` | Provider interfaces + `SearchResult` / `ProviderError`. |
| `src/providers/registry.py` | `@register` decorator → `REGISTRY`. |
| `src/providers/<type>.py` | One module per provider type. |
| `src/providers/pdf.py` | PDF detection + pypdf text extraction (used by the pipeline). |
| `src/pipeline_config.py` | In-code instances + pipeline order. |
| `src/pipeline.py` | Instance loader + search/read logic. |
| `src/rerank.py` | `JinaReranker` — post-merge rerank of search results. |
| `src/settings.py` | Non-secret knobs (pydantic-settings). |
| `src/server.py` | `build_server()` with the 3 `@mcp.tool` definitions. |
| `main.py` | Thin entry point: build server, run streamable-http. |
| `tests/` | pytest suite (network mocked with respx). |
