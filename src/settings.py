"""Server settings for the research-mcp facade.

Only NON-secret operational knobs live here, and every field has a default — so
``Settings()`` never fails on its own. Provider secrets/URLs are NOT declared
here: they are read by name (``os.getenv``) in the instance loader
(``src/pipeline.py``), keeping this model small and the provider list open.

Env var names map to field names case-insensitively (e.g. ``MCP_PORT`` →
``mcp_port``).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config_errors import load_settings_or_exit


class Settings(BaseSettings):
    # --- MCP transport -------------------------------------------------------
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    # --- Behaviour knobs -----------------------------------------------------
    log_level: str = "INFO"
    # Persistent log file (under data/ — survives restarts/image updates via the
    # mounted volume). Docker's json-file driver is rotation-capped, so a real
    # file is kept here for long-term, per-request log lines.
    log_file: str = "data/research-mcp.log"
    # loguru rotation/retention for the file sink (size trigger, age cutoff).
    log_rotation: str = "20 MB"
    log_retention: str = "14 days"
    # httpx request timeout in seconds for all outbound HTTP calls.
    request_timeout: float = 25.0
    # Read content shorter than this is "thin" → try the next read provider.
    fallback_min_chars: int = 400
    # Max number of concurrent page reads inside read_pages. (The per-call url
    # cap is a hard constant in server.py — see READ_PAGES_MAX — so the tool's
    # "up to 20" promise stays true regardless of environment overrides.)
    read_pages_concurrency: int = 5
    # Extra retry attempts for transient HTTP failures (per provider request).
    retries: int = 1

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# All fields have defaults, so this cannot fail on missing ENV; we still route
# through the helper for consistency with every other entrypoint.
settings = load_settings_or_exit(Settings)
