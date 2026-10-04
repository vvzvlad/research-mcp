"""Failure taxonomy: turn an exception into one short, model-facing category.

Pure and I/O-free (like ``src/formatting.py``): the only input is an exception —
its ``reason``, its type, and the causes chained behind it — and the only output
is one of the constants below. The pipeline tags every provider failure with
one, so the status lines can say *why* a page did not open instead of dumping
raw exception text at the model.

The category of our own failures is set where the error is raised, never
guessed from its message. Two steps, in this order:

1. A ``ProviderError`` whose raise site set ``reason`` (``src/providers/_http.py``
   for the HTTP status of a provider's own API, each provider for what it
   recognised in an answer) — that reason is the answer.
2. Otherwise the exception chain (``__cause__`` / ``__context__``, bounded like
   ``_is_tls_verify_error`` in ``src/pipeline.py``), which carries the real
   transport-level cause as third-party httpx / ssl / socket exceptions:
   timeout / TLS / DNS / other network. Nothing there → ``other``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import socket
import ssl

import httpx

# The categories. English in code (like every other identifier); the Russian
# labels the model sees live in src/formatting.py.
TIMEOUT = "timeout"
RATE_LIMIT = "rate-limit"
NO_CREDITS = "no-credits"
ACCESS_DENIED = "access-denied"
BOT_PROTECTION = "bot-protection"
# The page itself does not exist (the site answered 404/410): unlike every
# other category this is a fact about the url, not about the way we fetched it.
NOT_FOUND = "not-found"
TLS = "tls"
DNS = "dns"
NETWORK = "network"
EMPTY = "empty"
OTHER = "other"

# Bound the chain walk exactly like _is_tls_verify_error does: deep enough for
# httpx's wrapping, cheap, and immune to a self-referencing chain.
_MAX_CHAIN = 6

# What a failed name resolution says when it is not a socket.gaierror instance
# (e.g. it was already flattened into a message).
_DNS_TEXT = ("name or service not known", "nodename nor servname")


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` plus the causes behind it, up to ``_MAX_CHAIN`` links."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < _MAX_CHAIN:
        chain.append(current)
        # `raise X from None` hides the context on purpose; honour that.
        if current.__cause__ is not None or current.__suppress_context__:
            current = current.__cause__
        else:
            current = current.__context__
    return chain


def classify(exc: BaseException) -> str:
    """Return the failure category of ``exc`` — one of the constants above."""
    # Imported here, not at module level: src.providers imports every provider,
    # and the providers import this module.
    from src.providers.base import ProviderError

    if isinstance(exc, ProviderError) and exc.reason is not None:
        return exc.reason

    chain = _chain(exc)
    texts = [str(item) for item in chain]

    if any(isinstance(item, httpx.TimeoutException) for item in chain):
        return TIMEOUT
    if any(isinstance(item, ssl.SSLCertVerificationError) for item in chain) or any(
        "CERTIFICATE_VERIFY_FAILED" in text for text in texts
    ):
        return TLS
    if any(isinstance(item, socket.gaierror) for item in chain) or any(
        marker in text.lower() for text in texts for marker in _DNS_TEXT
    ):
        return DNS
    # Everything else httpx could not put on the wire: connect/read/proxy errors.
    if any(isinstance(item, httpx.TransportError) for item in chain):
        return NETWORK

    return OTHER


def for_target_status(code: int) -> str:
    """The category for "the TARGET site answered HTTP ``code``".

    Only for the status of the page being read, never for a provider's own API
    endpoint: a 404 there says nothing about the url.
    """
    if code in (404, 410):
        return NOT_FOUND
    if code in (401, 403):
        return ACCESS_DENIED
    if code == 429:
        return RATE_LIMIT
    return OTHER


def dominant_reason(failures: Sequence[tuple[str, str]]) -> str:
    """The one category that best describes a whole run of per-provider failures.

    ``failures`` is the ``(instance, reason)`` list a failed read collects. The
    most frequent reason wins, ties go to the earliest attempt — so the answer is
    deterministic — and an empty list (an error raised before any provider ran)
    is ``other``.
    """
    if not failures:
        return OTHER
    counts = Counter(reason for _, reason in failures)
    top = max(counts.values())
    for _, reason in failures:
        if counts[reason] == top:
            return reason
    return OTHER  # unreachable: `top` came from one of the reasons above
