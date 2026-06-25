"""Turn a pydantic-settings ValidationError into a clear, actionable startup message.

Reused by every entrypoint that builds settings — the app and any standalone
service (e.g. an MCP server) — so a missing or invalid environment variable fails
fast with a readable message (naming the env var) instead of a raw pydantic
traceback.
"""

import sys
from typing import Callable, TypeVar

from pydantic import ValidationError

T = TypeVar("T")


class ConfigError(Exception):
    """A non-pydantic startup configuration problem (e.g. no providers enabled).

    Carries a human-readable, multi-line message; ``exit_with_config_error``
    prints it in the same style as a missing-ENV error and exits.
    """


def exit_with_config_error(message: str) -> "None":
    """Print a clear configuration error to stderr and exit(1)."""
    lines = [
        "Configuration error in environment / .env:",
        f"  {message}",
        "",
        "Set the required variables in .env (see .env.example) and try again.",
    ]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(1)


def load_settings_or_exit(factory: Callable[[], T]) -> T:
    """Build a settings object via `factory` (e.g. a BaseSettings subclass).

    On a configuration ValidationError, print a clear message that names the
    offending environment variable(s) and exit(1) — no pydantic traceback. Any
    non-ValidationError is left to propagate unchanged.
    """
    try:
        return factory()
    except ValidationError as exc:
        missing: list[str] = []
        invalid: list[str] = []
        for err in exc.errors():
            # loc[0] is the field name; the env var is its upper-case form.
            name = str(err["loc"][0]).upper() if err.get("loc") else "?"
            if err.get("type") == "missing":
                missing.append(name)
            else:
                invalid.append(f"{name}: {err.get('msg')}")
        lines = ["Configuration error in environment / .env:"]
        if missing:
            lines.append("  Missing required variable(s):")
            lines.extend(f"    - {n}" for n in missing)
        if invalid:
            lines.append("  Invalid value(s):")
            lines.extend(f"    - {item}" for item in invalid)
        lines.append("")
        lines.append("Set them in .env (see .env.example) and try again.")
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(1)
