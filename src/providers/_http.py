"""Shared HTTP helpers for providers: transient retry + credit/limit handling.

Policy (cross-cutting, applied uniformly by every provider):

- Transient failures — ``httpx.TransportError`` / ``TimeoutException`` /
  ``RemoteProtocolError`` and HTTP 5xx — are retried ``retries`` extra times with
  a short backoff.
- ``402`` (out of credits) and ``429`` (rate limited) are treated as a hard
  provider failure → ``ProviderError`` (no retry). This is what makes a paid
  instance fail over to the next one (e.g. tavily-1 → tavily-2).
- Any other 4xx is also a ``ProviderError`` (the provider cannot serve this).
"""

from __future__ import annotations

import asyncio

import httpx

from src.providers.base import ProviderError

# Errors worth a quick retry — usually a blip, not a permanent condition.
_TRANSIENT_EXC = (
    httpx.TransportError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)

# Short fixed backoff between retries (seconds). Kept tiny on purpose.
_BACKOFF_SECONDS = 0.3


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retries: int,
    provider: str,
    **kwargs,
) -> httpx.Response:
    """Perform an HTTP request with the shared retry/credit policy.

    Returns a successful (2xx/3xx) ``httpx.Response``. Raises ``ProviderError``
    for credit/limit (402/429), other 4xx, exhausted retries, or transport
    errors. ``provider`` is used only for clearer error messages.
    """
    attempts = retries + 1
    last_error: str = "unknown error"
    for attempt in range(attempts):
        try:
            response = await client.request(method, url, **kwargs)
        except _TRANSIENT_EXC as exc:
            last_error = f"transport error: {exc}"
            if attempt + 1 < attempts:
                await asyncio.sleep(_BACKOFF_SECONDS)
                continue
            raise ProviderError(f"{provider}: {last_error}") from exc

        status = response.status_code
        if status in (402, 429):
            # Out of credits / rate limited — do NOT retry, fail over instead.
            reason = "out of credits" if status == 402 else "rate limited"
            raise ProviderError(f"{provider}: {reason} (HTTP {status})")
        if 500 <= status < 600:
            last_error = f"HTTP {status}"
            if attempt + 1 < attempts:
                await asyncio.sleep(_BACKOFF_SECONDS)
                continue
            raise ProviderError(f"{provider}: server error after retries (HTTP {status})")
        if 400 <= status < 500:
            raise ProviderError(f"{provider}: client error (HTTP {status})")
        return response

    # Unreachable, but keep the type checker and callers honest.
    raise ProviderError(f"{provider}: {last_error}")
