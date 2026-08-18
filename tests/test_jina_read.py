"""Jina Reader provider: token budget header and the keyed readerlm retry tier.

The keyed retry contract (verified against the docs 2026-08-18): a thin/empty
first answer escalates ONCE to ``X-Respond-With: readerlm-v2`` +
``X-Engine: browser``; the longer of the two texts wins; keyless mode never
retries (the heavy tier is billed). Network I/O is mocked with respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.jina import JINA_READER_BASE, JinaRead

URL = "https://doc.test/page"
READER_URL = f"{JINA_READER_BASE}{URL}"

# Comfortably over the fallback_min_chars=400 default used by make_config.
FULL_TEXT = "# Article\n\n" + ("Long extracted paragraph. " * 40)
THIN_TEXT = "# Stub\n\nAlmost nothing here."


# -- token budget header ---------------------------------------------------


@respx.mock
async def test_token_budget_header_sent_when_keyed_and_budget_set(make_config):
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=FULL_TEXT))
    provider = JinaRead(
        make_config("jina", api_key="k", options={"token_budget": "100000"})
    )
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    request = route.calls.last.request
    assert request.headers["X-Token-Budget"] == "100000"
    assert request.headers["Authorization"] == "Bearer k"
    assert request.headers["X-Return-Format"] == "markdown"


@respx.mock
async def test_token_budget_header_absent_when_keyless(make_config):
    # Keyless requests are not billed, so the budget header would only add a
    # failure mode for free — it must not be sent.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=FULL_TEXT))
    provider = JinaRead(make_config("jina", options={"token_budget": "100000"}))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    request = route.calls.last.request
    assert "X-Token-Budget" not in request.headers
    assert "Authorization" not in request.headers


@respx.mock
async def test_token_budget_header_absent_when_budget_zero(make_config):
    # JINA_TOKEN_BUDGET=0 is the documented way to disable the cap.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=FULL_TEXT))
    provider = JinaRead(make_config("jina", api_key="k", options={"token_budget": "0"}))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert "X-Token-Budget" not in route.calls.last.request.headers


# -- keyed readerlm retry tier ---------------------------------------------


@respx.mock
async def test_thin_first_answer_triggers_one_readerlm_retry(make_config):
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=THIN_TEXT),
        httpx.Response(200, text=FULL_TEXT),
    ]
    provider = JinaRead(
        make_config("jina", api_key="k", options={"token_budget": "100000"})
    )
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 2  # exactly one escalation, not a loop
    first, second = route.calls[0].request, route.calls[1].request
    # The first attempt is the plain cheap conversion...
    assert "X-Respond-With" not in first.headers
    assert "X-Engine" not in first.headers
    # ...and the retry adds the heavy tier while keeping the base headers.
    assert second.headers["X-Respond-With"] == "readerlm-v2"
    assert second.headers["X-Engine"] == "browser"
    assert second.headers["X-Return-Format"] == "markdown"
    assert second.headers["X-Token-Budget"] == "100000"


@respx.mock
async def test_full_first_answer_gets_no_retry(make_config):
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=FULL_TEXT))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 1  # good enough → no second (3x-priced) request


@respx.mock
async def test_longer_of_the_two_texts_wins(make_config):
    # Either tier can come back thin; when the retry extracts LESS than the
    # first attempt, the first answer is kept.
    shorter = "tiny"
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=THIN_TEXT),
        httpx.Response(200, text=shorter),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()
    assert route.call_count == 2


@respx.mock
async def test_retry_http_failure_falls_back_to_the_thin_first_text(make_config):
    # The heavy tier failing must not mask a usable thin first answer — the
    # pipeline can still use it as best_thin. The escalation goes out with
    # retries=0 regardless of config.retries=1 (the make_config default): the
    # fallback text is already in hand, so re-sending the slowest, 3x-priced
    # tier could only add latency — hence exactly TWO requests, not three.
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=THIN_TEXT),
        httpx.Response(500),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()
    assert route.call_count == 2  # one cheap attempt + ONE unretried escalation


@respx.mock
async def test_successful_escalation_emits_exactly_one_log_line(
    make_config, capture_logs
):
    # The provider's escalation log line is the accounting record for the
    # extra billed heavy-tier call (see the comment above the logger.info in
    # src/providers/jina.py) — a successful escalation must emit exactly one.
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=THIN_TEXT),
        httpx.Response(200, text=FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert sum("readerlm-v2 escalation" in m for m in capture_logs) == 1


@respx.mock
async def test_failed_escalation_request_emits_no_log_line(make_config, capture_logs):
    # A failed escalation request is not billed, so it must not leave the
    # accounting line either — counting the lines would overstate spend.
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=THIN_TEXT),
        httpx.Response(500),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()  # the thin fallback still comes through
    assert route.call_count == 2  # the escalation WAS attempted and failed
    assert not any("readerlm-v2 escalation" in m for m in capture_logs)


@respx.mock
async def test_both_attempts_empty_raise_empty_response(make_config):
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=""),
        httpx.Response(200, text="   "),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 2


@respx.mock
async def test_first_attempt_empty_and_retry_failing_propagates_the_retry_error(
    make_config,
):
    # With nothing to fall back to, the retry's HTTP failure is the real
    # story — it propagates instead of being swallowed into "empty response".
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=""),
        httpx.Response(404),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "HTTP 404" in str(excinfo.value)


@respx.mock
async def test_keyless_mode_never_retries(make_config):
    # The heavy tier is a paid feature; keyless just returns the thin text (the
    # pipeline treats it as best_thin) without a second request.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=THIN_TEXT))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()
    assert route.call_count == 1


@respx.mock
async def test_empty_body_with_zero_min_chars_still_raises(make_config):
    # fallback_min_chars=0 must not let "" through as a successful read: the
    # length check alone would accept it (len("") >= 0) and bypass the final
    # empty-guard. Keyless, so there is no escalation to muddy the picture.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=""))
    provider = JinaRead(make_config("jina", fallback_min_chars=0))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_keyed_empty_body_with_zero_min_chars_still_escalates(make_config):
    # The keyed twin of the test above: with fallback_min_chars=0 an empty
    # first body must not pass as success (the `text and` guard) — it counts
    # as thin/empty and escalates to the heavy tier, whose text is returned.
    route = respx.get(READER_URL)
    route.side_effect = [
        httpx.Response(200, text=""),
        httpx.Response(200, text=FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k", fallback_min_chars=0))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 2  # the empty first answer triggered the escalation


@respx.mock
async def test_keyless_empty_response_raises(make_config):
    route = respx.get(READER_URL).mock(return_value=httpx.Response(200, text=""))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_first_attempt_http_failure_gets_no_readerlm_retry(make_config):
    # An upstream 4xx means the site refused jina's fetch; the LM tier fixes
    # parsing, not access — so the failure propagates without escalation.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(403))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.read(client, URL)
    assert route.call_count == 1  # 4xx is not retried by the shared policy either
