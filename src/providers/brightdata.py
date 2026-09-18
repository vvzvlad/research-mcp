"""Bright Data Web Unlocker read provider.

API (verified 2026-09-18 against
https://docs.brightdata.com/api-reference/rest-api/unlocker/unlock-website and
https://docs.brightdata.com/scraping-automation/web-unlocker/send-your-first-request):
POST ``https://api.brightdata.com/request`` with header
``Authorization: Bearer {api_key}`` and body
``{"zone": zone, "url": url, "format": "raw", "data_format": "markdown"}``.

``format: "raw"`` means the response body IS the fetched content as plain text —
there is NO JSON envelope to unwrap and no ``success`` flag to check (the other
value, ``json``, would wrap it as ``{status_code, headers, body}``).
``data_format: "markdown"`` has Bright Data convert the page to Markdown before
returning it, which is exactly what this pipeline wants.

Role: the LAST link of the read chain. Unlike every provider before it, this one
solves anti-bot challenges and CAPTCHAs itself — rotating its own proxy pool and
retrying with different fingerprints — so it is what still reaches pages the
cheaper readers bounce off: marketplaces and Cloudflare-protected sites. It goes
last because it is the heaviest option, not because it is the least reliable.

Cost: the free tier covers 5000 requests per month, and unsuccessful requests
are never billed ("you are charged only for successful requests"), so an attempt
that fails here costs nothing but latency — which is part of why it is a safe
last resort.
"""

from __future__ import annotations

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register

BRIGHTDATA_REQUEST_ENDPOINT = "https://api.brightdata.com/request"


@register("brightdata")
class BrightDataUnlocker:
    """Read a page as Markdown via Bright Data Web Unlocker (api_key + zone)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("brightdata requires an api_key")
        # The zone name is an account-level setting — the Web Unlocker zone
        # created in the Bright Data control panel — not a secret, so it travels
        # as `token` (BRIGHTDATA_ZONE). It is a REQUIRED body field: without it
        # the API answers 400, so refuse to build the instance instead.
        if not config.token:
            raise ValueError("brightdata requires a zone")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        body = {
            "zone": self._config.token,
            "url": url,
            "format": "raw",
            "data_format": "markdown",
        }
        response = await request_with_retry(
            client,
            "POST",
            BRIGHTDATA_REQUEST_ENDPOINT,
            json=body,
            headers={"Authorization": f"Bearer {self._config.api_key}"},
            retries=self._config.retries,
            provider=self.name,
        )
        # With format: "raw" the body is the Markdown itself — response.text,
        # not response.json(). An empty body means the unlocker came back with
        # nothing usable, which is a failure for this provider.
        text = response.text.strip()
        if not text:
            raise ProviderError(f"{self.name}: empty response")
        return text
