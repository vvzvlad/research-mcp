"""Tavily Extract read provider: the spent-plan HTTP 432 signal.

Network mocked with respx, like the rest of the suite. The response-parsing and
failover behaviour of this provider is exercised through the pipeline tests;
this file pins the one thing unique to the read module — that it opts HTTP 432
into the shared out-of-credits branch.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src import failure_reason
from src.providers.base import ProviderError
from src.providers.tavily import TAVILY_EXTRACT_ENDPOINT, TavilyRead


@respx.mock
async def test_http_432_is_out_of_credits_not_a_client_error(make_config):
    # A spent Tavily plan answers extraction with the non-standard HTTP 432; the
    # read provider opts that status into the shared out-of-credits branch, so it
    # reads as out-of-credits, exhausts the quota, and does not retry.
    route = respx.post(TAVILY_EXTRACT_ENDPOINT).mock(
        return_value=httpx.Response(432, text="plan exhausted")
    )
    provider = TavilyRead(make_config("tavily", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, "https://doc.test/page")
    assert "out of credits (HTTP 432)" in str(excinfo.value)
    assert excinfo.value.reason == failure_reason.NO_CREDITS
    assert excinfo.value.exhausts_quota is True
    assert route.call_count == 1
