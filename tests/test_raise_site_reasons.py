"""The failure category each read provider sets where it raises.

The category is no longer read back from the message text, so a lost
``reason=`` would silently turn into ``other``; these tests pin the read-side
raise sites the old wording table used to cover.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src import failure_reason
from src.providers.base import ProviderError
from src.providers.brightdata import BRIGHTDATA_REQUEST_ENDPOINT, BrightDataUnlocker
from src.providers.crawl4ai import Crawl4aiRead
from src.providers.firecrawl import FIRECRAWL_SCRAPE_ENDPOINT, FirecrawlRead
from src.providers.tavily import TAVILY_EXTRACT_ENDPOINT, TavilyRead

URL = "https://example.com/article"


async def _reason(provider, url: str = URL) -> str | None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.read(client, url)
    return excinfo.value.reason


@respx.mock
async def test_tavily_page_not_found_is_not_found(make_config):
    # Measured live: tavily reports a missing page as "404 page not found".
    respx.post(TAVILY_EXTRACT_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"results": [], "failed_results": [{"url": URL, "error": "404 page not found"}]}
        )
    )
    reason = await _reason(TavilyRead(make_config("tavily-1", api_key="k")))
    assert reason == failure_reason.NOT_FOUND


@respx.mock
async def test_tavily_other_failed_result_has_no_reason(make_config):
    respx.post(TAVILY_EXTRACT_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json={"results": [], "failed_results": [{"url": URL, "error": "timeout"}]}
        )
    )
    assert await _reason(TavilyRead(make_config("tavily-1", api_key="k"))) is None


@respx.mock
async def test_tavily_empty_extraction_is_empty(make_config):
    respx.post(TAVILY_EXTRACT_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"results": [], "failed_results": []})
    )
    assert await _reason(TavilyRead(make_config("tavily-1", api_key="k"))) == failure_reason.EMPTY


@respx.mock
async def test_crawl4ai_empty_markdown_is_bot_protection(make_config):
    respx.post("http://crawl4ai.test/md").mock(
        return_value=httpx.Response(200, json={"markdown": "  ", "success": True})
    )
    provider = Crawl4aiRead(make_config("crawl4ai", url="http://crawl4ai.test", token="t"))
    assert await _reason(provider) == failure_reason.BOT_PROTECTION


@respx.mock
async def test_firecrawl_empty_markdown_is_empty(make_config):
    respx.post(FIRECRAWL_SCRAPE_ENDPOINT).mock(
        return_value=httpx.Response(200, json={"success": True, "data": {"markdown": ""}})
    )
    provider = FirecrawlRead(make_config("firecrawl", api_key="k"))
    assert await _reason(provider) == failure_reason.EMPTY


@respx.mock
async def test_brightdata_empty_body_is_empty(make_config):
    respx.post(BRIGHTDATA_REQUEST_ENDPOINT).mock(return_value=httpx.Response(200, text="  "))
    provider = BrightDataUnlocker(make_config("brightdata", api_key="k", token="zone"))
    assert await _reason(provider) == failure_reason.EMPTY
