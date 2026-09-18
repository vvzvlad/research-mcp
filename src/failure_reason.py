"""Failure taxonomy: turn an exception into one short, model-facing category.

Pure and I/O-free (like ``src/formatting.py``): the only input is an exception —
its type, its message, and the causes chained behind it — and the only output is
one of the constants below. The pipeline tags every provider failure with one,
so the status lines can say *why* a page did not open instead of dumping raw
exception text at the model.

Two passes, in this order:

1. The exception chain (``__cause__`` / ``__context__``, bounded like
   ``_is_tls_verify_error`` in ``src/pipeline.py``), which carries the real
   transport-level cause: timeout / TLS / DNS / other network.
2. Tolerant matching on OUR OWN ``ProviderError`` wording, built in
   ``src/providers/_http.py`` and the providers. Deliberately loose — a
   case-insensitive substring, or an ``HTTP <code>`` anywhere in the text — so a
   reworded provider message still lands in the right bucket.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import re
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
TLS = "tls"
DNS = "dns"
NETWORK = "network"
EMPTY = "empty"
OTHER = "other"

# Bound the chain walk exactly like _is_tls_verify_error does: deep enough for
# httpx's wrapping, cheap, and immune to a self-referencing chain.
_MAX_CHAIN = 6

# "HTTP 429", "HTTP429", "http 402" — the code may sit anywhere in the message.
_HTTP_STATUS = re.compile(r"http\s*(\d{3})", re.IGNORECASE)

# What a failed name resolution says when it is not a socket.gaierror instance
# (e.g. it was already flattened into a message).
_DNS_TEXT = ("name or service not known", "nodename nor servname")


def _chain(exc: BaseException) -> list[BaseException]:
    """``exc`` plus the causes behind it, up to ``_MAX_CHAIN`` links."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < _MAX_CHAIN:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify(exc: BaseException) -> str:
    """Return the failure category of ``exc`` — one of the constants above."""
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

    return _from_text(" | ".join(texts))


def _from_text(text: str) -> str:
    """Match our own ProviderError wording; ``other`` when nothing fits."""
    lowered = text.lower()
    codes = set(_HTTP_STATUS.findall(lowered))
    # "throttled" is our OWN most frequent search failure: searxng and brave skip
    # their turn when the local slot is taken. For the model that is the same
    # advice as a remote 429 — wait, do not retry now.
    if "rate limited" in lowered or "throttled" in lowered or "429" in codes:
        return RATE_LIMIT
    if "out of credits" in lowered or "402" in codes:
        return NO_CREDITS
    if codes & {"401", "403"}:
        return ACCESS_DENIED
    # Checked before `empty`: crawl4ai says "empty markdown (bot protection?)",
    # and the bot-protection guess is the more useful of the two signals.
    if "bot protection" in lowered:
        return BOT_PROTECTION
    # Every way a read provider says "there was nothing to take": trafilatura and
    # the pipeline's thin check, crawl4ai, jina ("empty response") and tavily
    # ("empty extraction"). They must agree, or dominant_reason splits the vote
    # of a page that is simply empty everywhere and answers "прочее".
    if any(
        marker in lowered
        for marker in (
            "content too thin",
            "no main content extracted",
            "empty markdown",
            "empty response",
            "empty extraction",
        )
    ):
        return EMPTY
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
