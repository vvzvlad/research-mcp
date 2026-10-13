"""YouTube video transcripts via the paid Supadata transcript API.

The ``supadata`` read provider (``SupadataRead``), a ``UrlSpecificReader`` that
accepts the same YouTube video urls as ``youtube`` and sits right behind it in
the read pipeline: it answers when YouTube bot-walls our egress IP. One call,
``GET /v1/transcript`` with ``mode=native`` — existing captions only, 1 credit
per video; the AI modes bill 2 credits per video minute. The transcript is
rendered with the ``youtube`` provider's timestamped paragraphs.
"""

from __future__ import annotations

import httpx

from src import failure_reason
from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register
from src.providers.youtube import _paragraphs, video_id

TRANSCRIPT_ENDPOINT = "https://api.supadata.ai/v1/transcript"

# Provider name for request_with_retry's error messages.
_PROVIDER = "supadata"


@register("supadata")
class SupadataRead:
    """Read a YouTube video url as its native captions, through Supadata."""

    # A generic reader gets the watch page chrome at best, and Google's captcha page from a flagged IP.
    falls_through = False

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    def accepts(self, url: str) -> bool:
        return video_id(url) is not None

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        """Fetch the transcript of ``url`` and render it as Markdown.

        Raises ``ProviderError`` when the video has no captions (206, worded as
        an empty response), on an async job (202, which native mode is not
        documented to start), or when the answer cannot be parsed or holds no
        segments.
        """
        response = await request_with_retry(
            client,
            "GET",
            TRANSCRIPT_ENDPOINT,
            retries=self._config.retries,
            provider=_PROVIDER,
            # The canonical watch url: Supadata documents that shape, while the
            # model may hand over /live/, /embed/ or youtu.be forms.
            params={
                "url": f"https://www.youtube.com/watch?v={video_id(url)}",
                "mode": "native",
                "text": "false",
            },
            headers={"x-api-key": self._config.api_key},
        )
        if response.status_code == 202:
            raise ProviderError("supadata: unexpected async job (HTTP 202)")
        if response.status_code == 206:
            raise ProviderError(
                "supadata: empty response (the video has no captions)",
                reason=failure_reason.EMPTY,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError(f"supadata: invalid response: {exc}") from exc
        if not isinstance(body, dict):
            raise ProviderError("supadata: invalid response: not a JSON object")

        content = body.get("content") or []
        if not content:
            raise ProviderError(
                "supadata: empty response (the transcript has no segments)",
                reason=failure_reason.EMPTY,
            )
        snippets = [(segment["offset"] / 1000, segment["text"]) for segment in content]
        lines = [
            f"# YouTube video {video_id(url)}",
            "",
            f"Captions: {body.get('lang', '')}",
            "",
            "## Transcript",
            "",
            "\n\n".join(_paragraphs(snippets)),
        ]
        return "\n".join(lines)
