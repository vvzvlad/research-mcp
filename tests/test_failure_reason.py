"""The failure classifier: an exception in, one category constant out.

Two families of inputs are pinned here: real transport exceptions (where the
answer comes from the __cause__/__context__ chain) and our own ProviderError
wording (where it comes from tolerant text matching).
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
    OTHER,
    RATE_LIMIT,
    TIMEOUT,
    TLS,
    classify,
    dominant_reason,
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


def test_chain_walk_is_bounded():
    # Deeper than the bound → the timeout at the bottom is not found (and the
    # classifier still answers instead of looping).
    exc: BaseException = httpx.ReadTimeout("timed out")
    for _ in range(8):
        exc = _wrapped("wrapper", exc)
    assert classify(exc) == OTHER


# -- our own ProviderError wording -----------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("tavily-1: rate limited (HTTP 429)", RATE_LIMIT),
        ("tavily-2: out of credits (HTTP 402)", NO_CREDITS),
        ("jina: client error (HTTP 403)", ACCESS_DENIED),
        ("jina: client error (HTTP 401)", ACCESS_DENIED),
        ("crawl4ai: empty markdown (bot protection?)", BOT_PROTECTION),
        ("firecrawl: empty markdown", EMPTY),
        ("trafilatura: no main content extracted", EMPTY),
        ("trafilatura: content too thin (12 chars)", EMPTY),
        # Every provider's own way of saying "nothing to take" must agree, or
        # dominant_reason splits the vote on a page that is empty everywhere.
        ("jina: empty response", EMPTY),
        ("tavily-1: empty extraction", EMPTY),
        # The local throttle is this project's most frequent search failure and
        # means exactly what a remote 429 means: wait, do not retry now.
        ("searxng: throttled (min interval 45s)", RATE_LIMIT),
        ("serper: server error after retries (HTTP 500)", OTHER),
        ("brave: unparseable response body", OTHER),
    ],
)
def test_provider_error_messages(message, expected):
    assert classify(ProviderError(message)) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # Reworded 4xx texts (another branch is rewriting them) must still land
        # in the right bucket: the code is matched anywhere, case-insensitively.
        ("tavily-1: не хватает кредитов, http402", NO_CREDITS),
        ("exa: refused by upstream, http 403", ACCESS_DENIED),
        ("brave: throttled, HTTP429", RATE_LIMIT),
        ("firecrawl: Rate Limited", RATE_LIMIT),
    ],
)
def test_message_matching_is_tolerant(message, expected):
    assert classify(ProviderError(message)) == expected


def test_unknown_exception_is_other():
    assert classify(ValueError("something odd")) == OTHER


# -- dominant_reason -------------------------------------------------------


def test_dominant_reason_picks_the_most_frequent():
    failures = [("trafilatura", NETWORK), ("jina", BOT_PROTECTION), ("crawl4ai", BOT_PROTECTION)]
    assert dominant_reason(failures) == BOT_PROTECTION


def test_dominant_reason_breaks_ties_by_first_attempt():
    assert dominant_reason([("trafilatura", NETWORK), ("jina", TIMEOUT)]) == NETWORK


def test_dominant_reason_without_failures_is_other():
    assert dominant_reason([]) == OTHER
