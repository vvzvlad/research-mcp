# Agent Instructions — research-mcp

An MCP "facade" that hides a pyramid of search/read providers behind a single
streamable-http MCP endpoint (`/mcp`) and exposes **4 tools**: `web_search`,
`read_page`, `read_pages`, `search_and_read`. No auth in the app: Traefik +
basicAuth on the host handles it.

## Plugin architecture (types + instances)
- A **type** is an implementation class (`src/providers/<type>.py`), registered
  with `@register("type")` → `REGISTRY` (`src/providers/registry.py`).
- An **instance** is a configured copy of a type whose secrets/URL come from
  **named ENV variables**. Several instances of one type are allowed.
- Which instances exist and the pipeline order live **in code**
  (`src/pipeline_config.py`: `INSTANCES`, `SEARCH_PIPELINE`, `READ_PIPELINE`);
  the loader in `src/pipeline.py` resolves the names with `os.getenv` and enables
  an instance only when its required variables are set.
- `duckduckgo` (search) and `trafilatura` (read) name no ENV var and are always
  on: the server must build and both tools must work on an empty environment.
- A `UrlSpecificReader` (`youtube`, `supadata`, `instagram`, `instagram_profile`) is offered
  only the urls it `accepts`, before the PDF probe; its answer is final, a
  failure falls through to the rest of the read chain.

## Project structure
- `src/server.py` — `build_server()` with the 4 `@mcp.tool` definitions.
- `src/pipeline.py` — instance loader, `ClientManager` (one httpx client per proxy
  URL), search (concurrent, merge/dedup/rerank), read (PDF probe + fallback chain)
  and `search_and_read`.
- `src/pipeline_config.py` — instances and pipeline order.
- `src/providers/` — `base.py` (interfaces, `ProviderError`, `ProviderConfig`),
  `_http.py` (shared retry and 402/429 → failover policy), `_url_guard.py` (SSRF
  guard), `pdf.py`, and one module per type.
- `src/rerank.py` — optional Jina rerank of merged search results.
- `src/quota.py` — persistent per-instance quotas and billed counters (SQLite).
- `src/failure_reason.py` — `classify(exc)` → failure category.
- `src/formatting.py` — pure renderers of the LLM-facing results and status lines.
- `src/settings.py` — non-secret knobs only, all defaulted.
- `main.py` — entry point: stderr + file log sink, build server, run.
- `data/` — runtime state (`research-mcp.log`, `quota.sqlite3`); gitignored,
  a docker volume in production.
- `tests/` — pytest, network mocked with respx.
- `server.json` — MCP registry metadata.

## Setup / test / run
```bash
make install           # create .venv and install dev/test deps
make env               # create .env from .env.example, then fill in the keys you have
make test              # run tests
make run               # serve streamable-http on MCP_HOST:MCP_PORT, endpoint /mcp
```

## CI and deploy
- `.gitea/workflows/tests.yml` runs the suite on pull requests and never pushes
  an image; `image-check-publish.yml` (push to
  `main`) tests, builds and pushes `gitea.vvzvlad.xyz/projects/research-mcp`
  (`:<sha>` + `:latest`). Production auto-updates from `:latest`.
- `docker-compose.yml` is a deploy template with placeholders only.

## Adding a provider
1. `src/providers/<type>.py` with an `@register("<type>")` class.
2. import it in `src/providers/__init__.py`.
3. add an `Instance(...)` (ENV var NAMES only) in `src/pipeline_config.py` and
   reference its name in a pipeline.
4. document the ENV var in `.env.example`.

## Invariants
- A search provider that did not run (throttled, blocked, broken markup) raises
  `ProviderError`; it never returns `[]`, which means "searched, found nothing".
- Local throttles (searxng, duckduckgo, brave) SKIP, never sleep:
  `Pipeline.search` awaits every provider.
- `ProviderError.reason` is set at the raise site; the category is never
  inferred from message text.
- Every url from the model passes the SSRF guard (entry check + redirect hook on
  the guarded clients); internal services use the plain clients.
- Per-instance proxies: `Instance.proxy_env` holds the ENV var NAME
  (`<INSTANCE>_PROXY`); `options_env` passes extra named vars (e.g. `GROQ_PROXY`)
  into `ProviderConfig.options`. Never log proxy URLs, keys or page bodies.

## Conventions
- Tool descriptions (LLM-facing) are in English and are the product of this
  project — do not change their wording. The status lines and failure labels in
  `src/formatting.py` are in Russian. All code/comments in English.
- `INSTANCES` holds ENV variable **names**, NEVER values. No real keys/urls in
  code, git, or `.env.example` (placeholders only); secrets live only in `.env` /
  prod `environment:`. Provider secrets are NOT Settings fields.
- Mutable state lives only in `data/`.
- All repeated actions go through `make` targets; Python runs from `.venv`.
- Tests are required for new code; in CI `build` depends on `test`.
- No `EXPOSE` in the Dockerfile — Traefik publishes via compose labels.

## Limits for agents
- Tests: at most three times the code (`wc -l` of `tests/` against the code). Over the limit,
  still write the test you need, and in the same change go through the old tests: drop or merge
  duplicates and tests of implementation rather than behaviour.
- Comments explain the code as it is and why it is built this way. History ("used to", "after
  the review", "since the incident") goes into the commit message.
- This file stays under ~150 lines: only what an agent would get wrong without it. No decision
  history, incident logs or retelling of the code.
