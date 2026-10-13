"""Supadata YouTube transcripts: url acceptance, the transcript call, the pipeline fallback.

All network I/O is mocked with respx.
"""

import httpx
import pytest
import respx

from src import failure_reason
from src.pipeline import Pipeline
from src.providers import _url_guard
from src.providers.base import ProviderError
from src.providers.supadata import TRANSCRIPT_ENDPOINT, SupadataRead
from src.providers.youtube import PLAYER_ENDPOINT
from tests.conftest import _clear_provider_env

VIDEO = "aA9R8bJAf68"
URL = f"https://www.youtube.com/watch?v={VIDEO}"

TRANSCRIPT = {
    "content": [
        {"text": "It's the first", "offset": 1920, "duration": 6520, "lang": "en"},
        {"text": "snippet here", "offset": 5120, "duration": 6559, "lang": "en"},
        {"text": "a minute later", "offset": 62500, "duration": 3000, "lang": "en"},
    ],
    "lang": "en",
    "availableLangs": ["en"],
}


@pytest.fixture
def reader(make_config):
    return SupadataRead(make_config("supadata", api_key="k"))


@pytest.mark.parametrize(
    ("url", "accepted"),
    [(URL, True), (f"https://youtu.be/{VIDEO}", True), ("https://example.com/article", False)],
)
def test_accepts_only_youtube_video_urls(reader, url, accepted):
    assert reader.accepts(url) is accepted


@respx.mock
async def test_read_renders_markdown(reader):
    route = respx.get(TRANSCRIPT_ENDPOINT).mock(return_value=httpx.Response(200, json=TRANSCRIPT))

    async with httpx.AsyncClient() as client:
        markdown = await reader.read(client, f"https://youtu.be/{VIDEO}")

    request = route.calls.last.request
    # Any accepted url shape goes out as the canonical watch url.
    assert request.url.params["url"] == URL
    assert request.url.params["mode"] == "native"
    assert request.url.params["text"] == "false"
    assert request.headers["x-api-key"] == "k"
    assert markdown.startswith(f"# YouTube video {VIDEO}\n")
    assert "Captions: en" in markdown
    assert markdown.endswith(
        "## Transcript\n\n[0:01] It's the first snippet here\n\n[1:02] a minute later"
    )


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(206, json={"error": "transcript-unavailable"}),
        httpx.Response(200, json={"content": [], "lang": "en", "availableLangs": []}),
    ],
    ids=["206", "empty-content"],
)
@respx.mock
async def test_read_without_captions_is_classified_empty(reader, answer):
    respx.get(TRANSCRIPT_ENDPOINT).mock(return_value=answer)

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await reader.read(client, URL)

    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


@respx.mock
async def test_read_async_job_is_an_error(reader):
    respx.get(TRANSCRIPT_ENDPOINT).mock(return_value=httpx.Response(202, json={"jobId": "j1"}))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError, match="unexpected async job"):
            await reader.read(client, URL)


def _public_dns(monkeypatch) -> None:
    """Resolve every host to a public address, so the SSRF guard needs no DNS."""

    async def _fake_resolve(host: str) -> list[str]:
        return ["142.250.74.46"]

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)


@respx.mock
async def test_pipeline_falls_back_to_supadata_when_youtube_is_bot_walled(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("SUPADATA_API_KEY", "k")
    bot_wall = {
        "playabilityStatus": {
            "status": "LOGIN_REQUIRED",
            "reason": "Sign in to confirm you’re not a bot",
        }
    }
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=bot_wall))
    respx.get(TRANSCRIPT_ENDPOINT).mock(return_value=httpx.Response(200, json=TRANSCRIPT))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "supadata"
    assert outcome.tried == ["youtube", "supadata"]
    assert outcome.failures == [("youtube", failure_reason.BOT_PROTECTION)]
    assert "[1:02] a minute later" in outcome.markdown
