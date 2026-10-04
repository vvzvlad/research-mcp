"""The failure classifier: an exception in, one category constant out.

Two families of inputs are pinned here: our own ProviderError (where the answer
is the ``reason`` its raise site set) and real transport exceptions (where it
comes from the __cause__/__context__ chain). The message text never decides.
"""

import socket
import ssl

import httpx
import pytest

from src.failure_reason import (
    ACCESS_DENIED,
    BOT_PROTECTION,
    DNS,
    EMPTY,
    NETWORK,
    NO_CREDITS,
    NOT_FOUND,
    OTHER,
    RATE_LIMIT,
    TIMEOUT,
    TLS,
    classify,
    dominant_reason,
    for_target_status,
)
from src.providers.base import ProviderError


def _wrapped(message: str, cause: BaseException) -> ProviderError:
    """A ProviderError raised from ``cause``, exactly like the providers do."""
    try:
        raise cause
    except BaseException as exc:  # noqa: BLE001 — building a chained exception
        error = ProviderError(message)
        error.__cause__ = exc
        return error


# -- the exception chain ---------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectTimeout("timed out"),
        httpx.PoolTimeout("timed out"),
    ],
)
def test_httpx_timeouts_are_timeout(exc):
    assert classify(exc) == TIMEOUT


def test_timeout_behind_our_provider_error():
    # _http.py wraps it as "transport error: ..." — the chain still decides.
    exc = _wrapped("tavily-1: transport error: timed out", httpx.ReadTimeout("timed out"))
    assert classify(exc) == TIMEOUT


def test_tls_certificate_failure():
    assert classify(ssl.SSLCertVerificationError("bad chain")) == TLS
    # httpx flattens it into a ConnectError message in practice.
    assert classify(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] verify failed")) == TLS


def test_dns_failure_by_type_and_by_text():
    assert classify(_wrapped("exa: transport error", socket.gaierror("nope"))) == DNS
    assert classify(httpx.ConnectError("Name or service not known")) == DNS
    assert classify(httpx.ConnectError("nodename nor servname provided")) == DNS


def test_other_transport_errors_are_network():
    assert classify(httpx.ConnectError("Connection refused")) == NETWORK
    assert classify(_wrapped("jina: transport error", httpx.ConnectError("reset"))) == NETWORK


def test_suppressed_context_is_not_followed():
    # `raise X from None` hides the context on purpose: the timeout behind it
    # must not decide the category.
    try:
        try:
            raise httpx.ReadTimeout("timed out")
        except httpx.ReadTimeout:
            raise ProviderError("jina: client error (HTTP 404)") from None
    except ProviderError as exc:
        assert classify(exc) == OTHER


def test_chain_walk_is_bounded():
    # Deeper than the bound → the timeout at the bottom is not found (and the
    # classifier still answers instead of looping).
    exc: BaseException = httpx.ReadTimeout("timed out")
    for _ in range(8):
        exc = _wrapped("wrapper", exc)
    assert classify(exc) == OTHER


# -- the reason set at the raise site --------------------------------------


@pytest.mark.parametrize(
    "reason",
    [RATE_LIMIT, NO_CREDITS, ACCESS_DENIED, BOT_PROTECTION, NOT_FOUND, EMPTY, OTHER],
)
def test_explicit_reason_is_returned(reason):
    assert classify(ProviderError("provider: whatever it says", reason=reason)) == reason


def test_explicit_reason_wins_over_the_chain():
    # jina reports a site's refusal even when a later ladder step timed out.
    exc = _wrapped("jina: target page returned HTTP 403", httpx.ReadTimeout("timed out"))
    exc.reason = ACCESS_DENIED
    assert classify(exc) == ACCESS_DENIED


def test_provider_error_without_reason_falls_to_the_chain():
    exc = _wrapped("exa: transport error: timed out", httpx.ConnectTimeout("timed out"))
    assert exc.reason is None
    assert classify(exc) == TIMEOUT


@pytest.mark.parametrize(
    "message",
    [
        # Worded like a category, but nobody set one: the text decides nothing.
        "tavily-1: rate limited (HTTP 429)",
        "crawl4ai: empty markdown (bot protection?)",
        "trafilatura: target page returned HTTP 404",
        "serper: server error after retries (HTTP 500)",
    ],
)
def test_provider_error_without_reason_or_chain_is_other(message):
    assert classify(ProviderError(message)) == OTHER


def test_unknown_exception_is_other():
    assert classify(ValueError("something odd")) == OTHER


# -- for_target_status -----------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (404, NOT_FOUND),
        (410, NOT_FOUND),
        (401, ACCESS_DENIED),
        (403, ACCESS_DENIED),
        (429, RATE_LIMIT),
        (400, OTHER),
        (402, OTHER),
        (500, OTHER),
        (503, OTHER),
    ],
)
def test_for_target_status(code, expected):
    assert for_target_status(code) == expected


# -- dominant_reason -------------------------------------------------------


def test_dominant_reason_picks_the_most_frequent():
    failures = [("trafilatura", NETWORK), ("jina", BOT_PROTECTION), ("crawl4ai", BOT_PROTECTION)]
    assert dominant_reason(failures) == BOT_PROTECTION


def test_dominant_reason_breaks_ties_by_first_attempt():
    assert dominant_reason([("trafilatura", NETWORK), ("jina", TIMEOUT)]) == NETWORK


def test_dominant_reason_without_failures_is_other():
    assert dominant_reason([]) == OTHER
