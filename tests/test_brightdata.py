"""Bright Data Web Unlocker provider: request shape and raw-body handling.

The contract under test (verified against the docs 2026-09-18): POST to
``https://api.brightdata.com/request`` with a Bearer key and a body carrying
``zone`` / ``url`` / ``format: "raw"`` / ``data_format: "markdown"``, whose
response body is the Markdown itself rather than a JSON envelope. Network I/O is
mocked with respx.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.brightdata import BRIGHTDATA_REQUEST_ENDPOINT, BrightDataUnlocker

URL = "https://shop.test/item/42"

MARKDOWN = "# Item 42\n\nPrice: 100\n"


def _provider(make_config, **kwargs):
    return BrightDataUnlocker(
        make_config("brightdata", api_key="k", token="web_unlocker1", **kwargs)
    )


@respx.mock
async def test_raw_markdown_body_is_returned(make_config):
    route = respx.post(BRIGHTDATA_REQUEST_ENDPOINT).mock(
        return_value=httpx.Response(200, text=MARKDOWN)
    )
    async with httpx.AsyncClient() as client:
        out = await _provider(make_config).read(client, URL)
    assert out == MARKDOWN.strip()
    assert route.call_count == 1


@respx.mock
async def test_request_shape(make_config):
    # format: "raw" + data_format: "markdown" is the whole point of this
    # provider — raw makes the body the content itself, markdown makes that
    # content Markdown. The zone comes from `token`, the key from `api_key`.
    route = respx.post(BRIGHTDATA_REQUEST_ENDPOINT).mock(
        return_value=httpx.Response(200, text=MARKDOWN)
    )
    async with httpx.AsyncClient() as client:
        await _provider(make_config).read(client, URL)
    request = route.calls.last.request
    assert str(request.url) == BRIGHTDATA_REQUEST_ENDPOINT
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Bearer k"
    assert json.loads(request.content) == {
        "zone": "web_unlocker1",
        "url": URL,
        "format": "raw",
        "data_format": "markdown",
    }


@respx.mock
async def test_empty_body_raises(make_config):
    # A blank body is not a usable read: there is no success flag in raw mode,
    # so the body being empty is the only signal that nothing came back.
    route = respx.post(BRIGHTDATA_REQUEST_ENDPOINT).mock(
        return_value=httpx.Response(200, text="   \n")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await _provider(make_config).read(client, URL)
    assert "empty response" in str(excinfo.value)
    assert route.call_count == 1


@respx.mock
async def test_out_of_credits_is_a_hard_failure(make_config):
    # 402 means the monthly allowance is gone; the shared policy turns it into a
    # ProviderError without retrying, so the pipeline fails over immediately.
    route = respx.post(BRIGHTDATA_REQUEST_ENDPOINT).mock(
        return_value=httpx.Response(402)
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await _provider(make_config).read(client, URL)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 is never retried


def test_requires_an_api_key(make_config):
    with pytest.raises(ValueError):
        BrightDataUnlocker(make_config("brightdata", token="web_unlocker1"))


def test_requires_a_zone(make_config):
    # The zone is a required body field; without it every request would 400.
    with pytest.raises(ValueError):
        BrightDataUnlocker(make_config("brightdata", api_key="k"))
