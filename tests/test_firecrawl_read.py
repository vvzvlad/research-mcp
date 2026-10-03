"""Firecrawl scrape: the site's error page behind a 200 is a failed read.

Firecrawl answers 200 and converts whatever the site returned; only
``data.metadata.statusCode`` says the site answered 404. Network I/O is mocked
with respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.failure_reason import NOT_FOUND, classify
from src.providers.base import ProviderError
from src.providers.firecrawl import FIRECRAWL_SCRAPE_ENDPOINT, FirecrawlRead

URL = "https://doc.test/page"
FULL_TEXT = "# Article\n\n" + ("Long extracted paragraph. " * 40)


def _scrape(markdown: str, status: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={"success": True, "data": {"markdown": markdown, "metadata": {"statusCode": status}}},
    )


@respx.mock
async def test_page_the_site_answered_404_is_a_failed_read(make_config):
    respx.post(FIRECRAWL_SCRAPE_ENDPOINT).mock(return_value=_scrape("# 404 Not found", 404))
    provider = FirecrawlRead(make_config("firecrawl", api_key="k"))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, URL)
    assert classify(excinfo.value) == NOT_FOUND


@respx.mock
async def test_page_the_site_answered_200_is_returned(make_config):
    respx.post(FIRECRAWL_SCRAPE_ENDPOINT).mock(return_value=_scrape(FULL_TEXT, 200))
    provider = FirecrawlRead(make_config("firecrawl", api_key="k"))
    async with httpx.AsyncClient() as client:
        out = await provider.read(client, URL)
    assert out == FULL_TEXT.strip()
