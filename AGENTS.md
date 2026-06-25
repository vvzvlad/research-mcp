# Agent Instructions — research-mcp

An MCP "facade" that hides a pyramid of search/read providers behind a single
streamable-http MCP endpoint and exposes **3 clean tools** with good Russian help
texts. No auth in the app: Traefik + basicAuth on the host handles it. No
application state — the only thing persisted is a log file under `data/` (kept on
a volume across restarts/image updates).

## Plugin architecture (types + instances)
- A **type** is an implementation class (`src/providers/<type>.py`), registered
  with `@register("type")` → `REGISTRY`.
- An **instance** is a configured copy of a type whose secrets/URL come from
  **named ENV variables**. Several instances of one type are allowed.
- Which instances exist and the pipeline order live **in code**
  (`src/pipeline_config.py`); keys/URLs come from ENV **by name**, resolved by
  `os.getenv` in `src/pipeline.py`.

## Project structure
- `src/providers/base.py` — `SearchProvider` / `ReadProvider` interfaces,
  `SearchResult`, `ProviderError`, `ProviderConfig`.
- `src/providers/registry.py` — `@register` decorator + `REGISTRY`.
- `src/providers/_http.py` — shared retry + 402/429 → failover policy.
- `src/providers/<type>.py` — searxng/serper/exa (search); trafilatura/jina/
  crawl4ai/tavily/firecrawl (read).
- `src/providers/pdf.py` — PDF detection + pypdf extraction (used by the pipeline, not a tool).
- `src/pipeline_config.py` — `INSTANCES`, `SEARCH_PIPELINE`, `READ_PIPELINE`.
- `src/pipeline.py` — instance loader + search (merge/dedup) and read (fallback) logic.
- `src/settings.py` — non-secret knobs only (all defaulted), incl. log file config.
- `src/server.py` — `build_server()` with the 3 `@mcp.tool` definitions.
- `main.py` — thin entry point: stderr + persistent file sink, build server, run streamable-http.
- `data/` — runtime state (persistent log file; gitignored, mounted as a volume).
- `tests/` — pytest (network mocked with respx).

## Logging
- stderr + a persistent file sink at `data/research-mcp.log` (loguru rotation +
  retention; survives restart/image update via the `data/` volume).
- `pipeline.search` / `pipeline.read` emit one per-request line each (tool,
  target url/query, winning provider/tier or `pdf`, count, latency, ok);
  `read_pages` adds a `count/ok` summary. Never log bodies or secrets.

## Setup / test / run
```bash
make install           # create .venv and install dev/test deps
cp .env.example .env   # fill in the keys you have  (shortcut: make env)
make test              # run tests
make run               # serve streamable-http on MCP_HOST:MCP_PORT, endpoint /mcp
```

## Adding a provider
1. `src/providers/<type>.py` with an `@register("<type>")` class.
2. import it in `src/providers/__init__.py`.
3. add an `Instance(...)` (ENV var NAMES only) in `src/pipeline_config.py` and
   reference its name in a pipeline.
4. document the ENV var in `.env.example`.

## Conventions
- Tool descriptions (LLM-facing) are in Russian and are the product of this
  project — do not change their wording. All other code/comments in English.
- `INSTANCES` holds ENV variable **names**, NEVER values. No real keys/urls in
  code, git, or `.env.example` (placeholders only); secrets live only in `.env` /
  prod `environment:`.
- Provider secrets are NOT Settings fields — read by name in the loader.
- The only runtime state is the log file under `data/` (golden rule: mutable
  state lives in `data/`, which is gitignored and a docker volume).
- All repeated actions go through `make` targets; Python runs from `.venv`.
- Tests are required for new code; in CI `build` depends on `test`.
- No `EXPOSE` in the Dockerfile — Traefik publishes via compose labels.
