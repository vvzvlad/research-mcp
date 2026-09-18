"""Shared HTTP policy: an exhausted balance reported with a non-402 status.

Serper answers an empty account with HTTP 400
``{"message":"Not enough credits","statusCode":400}`` rather than the
conventional 402 (verified live 2026-09-18: ``GET /account`` returns
``balance: 0`` while every search returns that 400). For months this surfaced as
``client error (HTTP 400)`` — an unpaid account that reads like a broken API.

These tests pin the classification and, just as importantly, that it stays a
single non-retried attempt: a billing state must fail over immediately, exactly
like every other 4xx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers._http import request_with_retry
from src.providers.base import ProviderError

URL = "https://provider.test/search"

# Verbatim shape of the serper body that prompted this policy.
SERPER_BODY = {"message": "Not enough credits", "statusCode": 400}


async def _call(retries: int = 1) -> httpx.Response:
    async with httpx.AsyncClient() as client:
        return await request_with_retry(
            client, "GET", URL, retries=retries, provider="probe"
        )


@respx.mock
async def test_serper_style_400_is_reported_as_out_of_credits():
    route = respx.get(URL).mock(return_value=httpx.Response(400, json=SERPER_BODY))

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "out of credits (HTTP 400)" in str(excinfo.value)
    # The status stays in the message: an operator needs to see that this was a
    # 400, not a real 402, when checking the vendor's dashboard.
    assert "client error" not in str(excinfo.value)
    # Billing state → no retry, exactly like 402/429.
    assert route.call_count == 1


@respx.mock
async def test_octen_style_403_insufficient_balance_is_out_of_credits():
    # Octen documents HTTP 403 "Insufficient balance in account" for the same
    # condition — a different status and a different wording from serper's.
    route = respx.get(URL).mock(
        return_value=httpx.Response(403, text="Insufficient balance in account")
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "out of credits (HTTP 403)" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_marker_match_is_case_insensitive():
    route = respx.get(URL).mock(
        return_value=httpx.Response(400, json={"error": "NOT ENOUGH CREDITS"})
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_unrelated_4xx_is_still_a_client_error():
    # The whole point of matching specific phrases: a genuinely malformed
    # request must keep reading as our bug, not as an unpaid invoice.
    route = respx.get(URL).mock(
        return_value=httpx.Response(400, json={"message": "q parameter is required"})
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "client error (HTTP 400)" in str(excinfo.value)
    assert "out of credits" not in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_the_word_credits_alone_does_not_trigger_the_classification():
    # A vendor explaining its pricing inside an unrelated error must not be
    # mistaken for an empty account — this is why the markers are phrases.
    route = respx.get(URL).mock(
        return_value=httpx.Response(
            400, json={"message": "This endpoint costs 2 credits per call"}
        )
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "client error (HTTP 400)" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_plain_402_still_reports_out_of_credits():
    # Unchanged behaviour, pinned so the new branch cannot swallow the old one.
    route = respx.get(URL).mock(return_value=httpx.Response(402, text="payment required"))

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "out of credits (HTTP 402)" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_429_stays_rate_limited_even_when_the_body_mentions_credits():
    # Rate limiting and an empty balance are different operational problems:
    # one clears by itself, the other needs a payment. 429 must keep its own
    # wording whatever the body says.
    route = respx.get(URL).mock(
        return_value=httpx.Response(429, json={"message": "Not enough credits"})
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call()

    assert "rate limited (HTTP 429)" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_credit_body_on_a_5xx_is_untouched_and_still_retries():
    # 5xx keeps the transient path regardless of body text: the marker check
    # lives in the 4xx branch only.
    route = respx.get(URL).mock(
        return_value=httpx.Response(500, json={"message": "Not enough credits"})
    )

    with pytest.raises(ProviderError) as excinfo:
        await _call(retries=1)

    assert "server error after retries (HTTP 500)" in str(excinfo.value)
    assert route.call_count == 2
