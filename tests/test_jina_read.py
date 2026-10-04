"""Jina Reader provider: token budget header, JSON answers and the escalation ladder.

Every request asks for jina's JSON envelope (``Accept: application/json``); the
Markdown is ``data.content`` and the site's refusal is read from
``data.httpStatus`` / ``data.warning`` only (shape measured live 2026-10-04).

The keyed escalation contract (verified against the docs 2026-08-18): a
thin/empty answer climbs a ladder of ever dearer tiers, one step at a time, and
stops as soon as a step returns enough text —

1. ``X-Respond-With: readerlm-v2`` + ``X-Engine: browser`` (3x tokens),
2. the same plus ``x-proxy: auto``, jina's residential pool (5x),
3. ``X-Respond-With: jina-ocr-v1`` in place of readerlm-v2 (40x), .pdf ONLY.

A missing page or a CAPTCHA wall fails at once; any other refusal of the plain
answer goes straight to the residential steps; a refusal on any step ends the
ladder.

The longest text of all attempts wins; a step that fails stops the ladder; each
step logs its own accounting line, and only after jina's JSON came back; keyless
mode never escalates (every step is billed). Every test pins
``route.call_count``: without it a test still passes with a whole step deleted.
Network I/O is mocked with respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.failure_reason import ACCESS_DENIED, BOT_PROTECTION, NOT_FOUND, classify
from src.providers.base import ProviderError
from src.providers.jina import JINA_READER_BASE, JinaRead
from tests.conftest import _jina_answer

URL = "https://doc.test/page"
READER_URL = f"{JINA_READER_BASE}{URL}"

# The OCR step is reachable only from a .pdf url.
PDF_URL = "https://doc.test/paper.pdf"
PDF_READER_URL = f"{JINA_READER_BASE}{PDF_URL}"

# Comfortably over the fallback_min_chars=400 default used by make_config.
FULL_TEXT = "# Article\n\n" + ("Long extracted paragraph. " * 40)
THIN_TEXT = "# Stub\n\nAlmost nothing here."
# A second thin answer, distinct from THIN_TEXT and longer, so "longest wins"
# stays observable while the ladder keeps climbing.
THIN_TEXT_2 = "# Stub\n\nAlmost nothing here either, honestly."


# -- token budget header ---------------------------------------------------


@respx.mock
async def test_token_budget_header_sent_when_keyed_and_budget_set(make_config):
    route = respx.get(READER_URL).mock(return_value=_jina_answer(FULL_TEXT))
    provider = JinaRead(
        make_config("jina", api_key="k", options={"token_budget": "100000"})
    )
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 1
    request = route.calls.last.request
    assert request.headers["X-Token-Budget"] == "100000"
    assert request.headers["Authorization"] == "Bearer k"
    assert request.headers["X-Return-Format"] == "markdown"


@respx.mock
async def test_token_budget_header_absent_when_keyless(make_config):
    # Keyless requests are not billed, so the budget header would only add a
    # failure mode for free — it must not be sent.
    route = respx.get(READER_URL).mock(return_value=_jina_answer(FULL_TEXT))
    provider = JinaRead(make_config("jina", options={"token_budget": "100000"}))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 1
    request = route.calls.last.request
    assert "X-Token-Budget" not in request.headers
    assert "Authorization" not in request.headers


@respx.mock
async def test_token_budget_header_absent_when_budget_zero(make_config):
    # JINA_TOKEN_BUDGET=0 is the documented way to disable the cap.
    route = respx.get(READER_URL).mock(return_value=_jina_answer(FULL_TEXT))
    provider = JinaRead(make_config("jina", api_key="k", options={"token_budget": "0"}))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 1
    assert "X-Token-Budget" not in route.calls.last.request.headers


# -- the JSON envelope -----------------------------------------------------


@respx.mock
async def test_every_request_asks_for_the_json_envelope(make_config):
    # The refusal lives in data.httpStatus / data.warning, which only the JSON
    # answer has — so every request, the plain one and each ladder step alike,
    # must ask for it.
    route = respx.get(PDF_READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer("tiny"),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, PDF_URL)
    assert route.call_count == 4
    assert all(
        call.request.headers["Accept"] == "application/json" for call in route.calls
    )


@respx.mock
async def test_keyless_request_asks_for_the_json_envelope(make_config):
    route = respx.get(READER_URL).mock(return_value=_jina_answer(FULL_TEXT))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 1
    assert route.calls.last.request.headers["Accept"] == "application/json"


@pytest.mark.parametrize(
    "response",
    [
        # The plain-text answer, as if the Accept header had been ignored.
        httpx.Response(200, text="Title: Page\n\nMarkdown Content:\n# Page"),
        # JSON, but no data object to read the content from.
        httpx.Response(200, json={"code": 200, "status": 20000}),
    ],
    ids=["not-json", "no-data-object"],
)
@respx.mock
async def test_invalid_json_answer_fails(make_config, response):
    route = respx.get(READER_URL).mock(return_value=response)
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "invalid JSON response" in str(excinfo.value)
    assert route.call_count == 1


# -- keyed readerlm retry tier ---------------------------------------------


@respx.mock
async def test_thin_first_answer_triggers_one_readerlm_retry(make_config):
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(FULL_TEXT),
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
    # The residential pool belongs to the NEXT step up: neither of these two
    # requests may pay 5x for a proxy nobody asked for yet.
    assert "x-proxy" not in first.headers
    assert "x-proxy" not in second.headers


@respx.mock
async def test_full_first_answer_gets_no_retry(make_config):
    route = respx.get(READER_URL).mock(return_value=_jina_answer(FULL_TEXT))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 1  # good enough → no second (3x-priced) request


@respx.mock
async def test_longest_of_all_attempts_wins(make_config):
    # Any tier can come back thin; when every escalation extracts LESS than the
    # first attempt, the first answer is the one kept. Non-pdf, so the ladder
    # ends after the residential step — three requests in total.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer("tiny"),
        _jina_answer("tinier"),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()
    assert route.call_count == 3


@respx.mock
async def test_retry_http_failure_falls_back_to_the_thin_first_text(make_config):
    # The heavy tier failing must not mask a usable thin first answer — the
    # pipeline can still use it as best_thin. The escalation goes out with
    # retries=0 regardless of config.retries=1 (the make_config default): the
    # fallback text is already in hand, so re-sending the slowest, 3x-priced
    # tier could only add latency — hence exactly TWO requests, not three.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
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
        _jina_answer(THIN_TEXT),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 2
    assert sum("readerlm-v2 escalation" in m for m in capture_logs) == 1


@respx.mock
async def test_failed_escalation_request_emits_no_log_line(make_config, capture_logs):
    # A failed escalation request is not billed, so it must not leave the
    # accounting line either — counting the lines would overstate spend.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        httpx.Response(500),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT.strip()  # the thin fallback still comes through
    assert route.call_count == 2  # the escalation WAS attempted and failed
    assert not any("readerlm-v2 escalation" in m for m in capture_logs)


@respx.mock
async def test_all_attempts_empty_raise_empty_response(make_config):
    # Every step answered with nothing in it: the ladder is exhausted (a
    # non-pdf url stops before the OCR step) and there is no text to hand back.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(""),
        _jina_answer("   "),
        _jina_answer("\n"),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 3


@respx.mock
async def test_first_attempt_empty_and_retry_failing_propagates_the_retry_error(
    make_config,
):
    # With nothing to fall back to, the retry's HTTP failure is the real
    # story — it propagates instead of being swallowed into "empty response".
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(""),
        httpx.Response(404),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "HTTP 404" in str(excinfo.value)
    assert route.call_count == 2  # a failed step ends the ladder, dearer steps too


@respx.mock
async def test_keyless_mode_never_retries(make_config):
    # The heavy tier is a paid feature; keyless just returns the thin text (the
    # pipeline treats it as best_thin) without a second request.
    route = respx.get(READER_URL).mock(return_value=_jina_answer(THIN_TEXT))
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
    route = respx.get(READER_URL).mock(return_value=_jina_answer(""))
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
        _jina_answer(""),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k", fallback_min_chars=0))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 2  # the empty first answer triggered the escalation


@respx.mock
async def test_keyless_empty_response_raises(make_config):
    route = respx.get(READER_URL).mock(return_value=_jina_answer(""))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_first_attempt_http_failure_gets_no_readerlm_retry(make_config):
    # jina's OWN endpoint answering 4xx is jina's failure, not the site's; the
    # LM tier fixes parsing, not that — so the failure propagates without
    # escalation.
    route = respx.get(READER_URL).mock(return_value=httpx.Response(403))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await provider.read(client, URL)
    assert route.call_count == 1  # 4xx is not retried by the shared policy either


# -- the residential-proxy step (x-proxy: auto) ----------------------------


@respx.mock
async def test_still_thin_after_readerlm_escalates_to_the_residential_proxy(
    make_config,
):
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(
        make_config("jina", api_key="k", options={"token_budget": "100000"})
    )
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 3  # plain → readerlm-v2 → residential proxy
    first, second, third = (route.calls[i].request for i in range(3))
    # x-proxy appears on the THIRD attempt and nowhere earlier — that is the
    # whole point of it being a separate (5x-priced) step.
    assert "x-proxy" not in first.headers
    assert "x-proxy" not in second.headers
    assert third.headers["x-proxy"] == "auto"
    # It is the readerlm tier plus a residential exit, not a different tier.
    assert third.headers["X-Respond-With"] == "readerlm-v2"
    assert third.headers["X-Engine"] == "browser"
    assert third.headers["X-Return-Format"] == "markdown"
    assert third.headers["X-Token-Budget"] == "100000"


@respx.mock
async def test_enough_text_from_readerlm_stops_before_the_residential_proxy(
    make_config,
):
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 2  # the ladder stops as soon as a step suffices


@respx.mock
async def test_successful_residential_escalation_emits_exactly_one_log_line(
    make_config, capture_logs
):
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        await provider.read(client, URL)
    assert route.call_count == 3
    assert sum("residential-proxy escalation" in m for m in capture_logs) == 1


@respx.mock
async def test_failed_residential_escalation_emits_no_log_line(
    make_config, capture_logs
):
    # A failed step is not billed, so it must not leave the accounting line —
    # and it stops the ladder, so no further (dearer) step is bought either.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        httpx.Response(500),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT_2.strip()  # the best text so far still comes through
    assert route.call_count == 3  # attempted once, unretried, then the ladder ends
    assert sum("readerlm-v2 escalation" in m for m in capture_logs) == 1
    assert not any("residential-proxy escalation" in m for m in capture_logs)


# -- the OCR step (jina-ocr-v1, .pdf only) ---------------------------------


@respx.mock
async def test_pdf_url_escalates_to_the_ocr_tier(make_config, capture_logs):
    route = respx.get(PDF_READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer("tiny"),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, PDF_URL)
    assert out == FULL_TEXT.strip()
    # Four sequential requests on a stubborn .pdf — the accepted worst case.
    assert route.call_count == 4
    fourth = route.calls[3].request
    assert fourth.headers["X-Respond-With"] == "jina-ocr-v1"  # replaces readerlm-v2
    assert fourth.headers["X-Engine"] == "browser"
    assert fourth.headers["x-proxy"] == "auto"
    assert fourth.headers["X-Return-Format"] == "markdown"
    # Every step logged its own line, exactly once, after its own answer.
    assert sum("readerlm-v2 escalation" in m for m in capture_logs) == 1
    assert sum("residential-proxy escalation" in m for m in capture_logs) == 1
    assert sum("jina-ocr-v1 escalation" in m for m in capture_logs) == 1


@respx.mock
async def test_non_pdf_url_never_reaches_the_ocr_tier(make_config, capture_logs):
    # 40x tokens is worth it for a scan and for nothing else, so a non-pdf url
    # ends the ladder after the residential step even when still thin.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer("tiny"),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == THIN_TEXT_2.strip()
    assert route.call_count == 3  # no fourth request
    assert not any(
        "jina-ocr-v1" in request.headers.get("X-Respond-With", "")
        for request in (call.request for call in route.calls)
    )
    assert not any("jina-ocr-v1 escalation" in m for m in capture_logs)


@respx.mock
async def test_pdf_detection_is_case_insensitive(make_config):
    upper_url = "https://doc.test/PAPER.PDF"
    route = respx.get(f"{JINA_READER_BASE}{upper_url}")
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, upper_url)
    assert out == FULL_TEXT.strip()
    assert route.call_count == 4
    assert route.calls[3].request.headers["X-Respond-With"] == "jina-ocr-v1"


@respx.mock
async def test_pdf_detection_looks_at_the_path_not_the_query(make_config):
    # ".pdf" in a query string names a parameter value, not the document this
    # url serves — OCR must not be bought for it.
    query_url = "https://doc.test/viewer?file=report.pdf"
    route = respx.get(f"{JINA_READER_BASE}{query_url}")
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer("tiny"),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, query_url)
    assert out == THIN_TEXT_2.strip()
    assert route.call_count == 3


@respx.mock
async def test_failed_ocr_escalation_emits_no_log_line(make_config, capture_logs):
    route = respx.get(PDF_READER_URL)
    route.side_effect = [
        _jina_answer(THIN_TEXT),
        _jina_answer(THIN_TEXT_2),
        _jina_answer("tiny"),
        httpx.Response(500),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, PDF_URL)
    assert out == THIN_TEXT_2.strip()  # the best text so far survives the failure
    assert route.call_count == 4  # the OCR step ran once, unretried
    assert not any("jina-ocr-v1 escalation" in m for m in capture_logs)


@respx.mock
async def test_keyless_pdf_never_escalates(make_config):
    # Every step of the ladder is billed, so a keyless instance climbs none of
    # it — not even on a .pdf.
    route = respx.get(PDF_READER_URL).mock(return_value=_jina_answer(THIN_TEXT))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, PDF_URL)
    assert out == THIN_TEXT.strip()
    assert route.call_count == 1


# -- the site's refusal inside a 200 ---------------------------------------
# jina answers 200 and reports the refusal only in data.httpStatus and
# data.warning; the refusal pages below are longer than fallback_min_chars, so
# length alone accepts them.


def _refused(http_status: int, warning: str | None = None) -> httpx.Response:
    return _jina_answer(
        "# Blocked\n\n" + "Access denied. " * 40,
        http_status=http_status,
        warning=warning,
    )


def _captcha() -> httpx.Response:
    # The scispace.com CAPTCHA wall as measured live: a 405 whose warning names
    # the CAPTCHA.
    return _refused(
        405,
        "Target URL returned error 405: Method Not Allowed\nThis page maybe "
        "requiring CAPTCHA, please make sure you are authorized to access this page.",
    )


@respx.mock
async def test_captcha_wall_fails_without_escalation(make_config):
    route = respx.get(READER_URL).mock(return_value=_captcha())
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "bot protection" in str(excinfo.value)
    assert classify(excinfo.value) == BOT_PROTECTION
    assert route.call_count == 1  # the residential exit does not break challenges


@pytest.mark.parametrize("status", [404, 410])
@respx.mock
async def test_missing_page_fails_without_escalation(make_config, status):
    route = respx.get(READER_URL).mock(
        return_value=_refused(status, f"Target URL returned error {status}: Not Found")
    )
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert f"HTTP {status}" in str(excinfo.value)
    assert classify(excinfo.value) == NOT_FOUND
    assert route.call_count == 1


@respx.mock
async def test_blocked_page_goes_straight_to_the_residential_exit(make_config):
    route = respx.get(READER_URL)
    route.side_effect = [
        _refused(403),
        _jina_answer(FULL_TEXT),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
    # plain → residential proxy: the parsing-only readerlm step is skipped.
    assert route.call_count == 2
    assert route.calls.last.request.headers["x-proxy"] == "auto"


@respx.mock
async def test_blocked_pdf_on_the_residential_exit_never_buys_ocr(make_config):
    route = respx.get(PDF_READER_URL).mock(return_value=_refused(403))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, PDF_URL)
    assert classify(excinfo.value) == ACCESS_DENIED
    assert route.call_count == 2  # plain → residential; OCR shares that exit


@respx.mock
async def test_refusal_survives_a_failing_residential_step(make_config):
    # The refusal met first is what the error names, not the later step's
    # timeout.
    route = respx.get(READER_URL)
    route.side_effect = [
        _refused(403),
        httpx.ReadTimeout("timed out"),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "HTTP 403" in str(excinfo.value)
    assert classify(excinfo.value) == ACCESS_DENIED
    assert route.call_count == 2


@respx.mock
async def test_captcha_on_a_parsing_step_stops_before_the_residential_exit(
    make_config,
):
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(""),
        _captcha(),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert classify(excinfo.value) == BOT_PROTECTION
    assert route.call_count == 2  # plain → readerlm-v2; no 5x residential step


@respx.mock
async def test_refusal_on_a_ladder_step_ends_the_ladder(make_config):
    # An empty plain answer climbs; the browser engine then meets the site's
    # refusal, which ends the ladder — no residential step is bought after it —
    # and is what the error names.
    route = respx.get(READER_URL)
    route.side_effect = [
        _jina_answer(""),
        _refused(403),
    ]
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "HTTP 403" in str(excinfo.value)
    assert classify(excinfo.value) == ACCESS_DENIED
    assert route.call_count == 2


@respx.mock
async def test_keyless_refusal_fails_with_the_target_status(make_config):
    route = respx.get(READER_URL).mock(return_value=_refused(403))
    provider = JinaRead(make_config("jina"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert "HTTP 403" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_error_quoted_in_the_article_body_is_not_a_refusal(make_config):
    # The refusal is read from the fields only: a 200 page whose Markdown quotes
    # jina's warning text is still the article.
    content = "Warning: Target URL returned error 404: quoted in a how-to.\n\n" + FULL_TEXT
    route = respx.get(READER_URL).mock(return_value=_jina_answer(content))
    provider = JinaRead(make_config("jina", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == content.strip()
    assert route.call_count == 1
