# research-mcp

A stateless **MCP facade** that hides a pyramid of search/read providers behind a
single streamable-http MCP endpoint and exposes just **3 clean tools** with good
Russian help texts. An LLM gets a simple "search → read" toolset; behind it,
several providers are tried, merged, and failed over automatically.

The app does **no authentication** — it is published through Traefik + basicAuth
on the host. It holds no application state: the only thing persisted is a log
file under `data/` (kept on a volume).

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

- **Search pipeline** (`searxng → serper → exa`): enabled instances run
  concurrently; results are merged and deduplicated by normalized URL (earlier
  pipeline position wins), then trimmed to `num_results`.
- **Read pipeline** (`trafilatura → jina → crawl4ai → tavily-1 → tavily-2 →
  firecrawl`): a single probe GET classifies the url. PDFs (Content-Type /
  `.pdf` / `%PDF` magic) are extracted with pypdf; for HTML, that same body is
  handed to `trafilatura` so the hot path never GETs twice, then the remaining
  instances are tried in order and the first to return content
  `>= FALLBACK_MIN_CHARS` wins.

Cross-cutting: one transient retry (5xx / transport errors) with a short backoff;
**402 (out of credits) / 429 (rate limited) are treated as a provider failure →
next instance** (this is what makes `tavily-1 → tavily-2` fail over).

An instance is **enabled** only if its required env var(s) are set; otherwise it
is skipped with a log line. `trafilatura` needs no config (always on); `jina`
works keyless (its key is optional). At startup the server requires at least one
search and one read instance, else it exits with a clear message.

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
`FALLBACK_MIN_CHARS`, `READ_PAGES_CONCURRENCY`, `RETRIES`. The `read_pages`
per-call url cap is a fixed `20` (hard constant, matching the tool description) —
not configurable.

Provider env vars: `SEARXNG_URL`, `SERPER_API_KEY`, `EXA_API_KEY`, `JINA_API_KEY`
(optional), `CRAWL4AI_URL` + `CRAWL4AI_TOKEN`, `TAVILY_1_API_KEY`,
`TAVILY_2_API_KEY`, `FIRECRAWL_API_KEY`.

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

CI builds the image and pushes it to `ghcr.io` (`test` → `build`, tags `latest` +
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
| `src/settings.py` | Non-secret knobs (pydantic-settings). |
| `src/server.py` | `build_server()` with the 3 `@mcp.tool` definitions. |
| `main.py` | Thin entry point: build server, run streamable-http. |
| `tests/` | pytest suite (network mocked with respx). |
