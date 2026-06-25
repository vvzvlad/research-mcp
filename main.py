"""Thin entry point: build the research-mcp facade and serve it over HTTP.

Transport is streamable-http on ``mcp_host:mcp_port`` (endpoint ``/mcp``). The
app does no auth — Traefik + basicAuth on the host handles that.
"""

import sys

from loguru import logger

from src.config_errors import ConfigError, exit_with_config_error
from src.server import build_server
from src.settings import settings


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    # Persistent file sink on the data/ volume: long-term, per-request log lines
    # that survive container restarts and image updates. enqueue=True keeps the
    # async hot path from blocking on disk I/O.
    logger.add(
        settings.log_file,
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    logger.info(
        "Starting research-mcp on {host}:{port}/mcp (streamable-http)",
        host=settings.mcp_host,
        port=settings.mcp_port,
    )
    try:
        server = build_server(settings)
    except ConfigError as exc:
        # No search/read provider enabled → clear message + exit(1), no traceback.
        exit_with_config_error(str(exc))
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
