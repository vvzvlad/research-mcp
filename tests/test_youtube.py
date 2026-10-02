"""YouTube transcripts: url detection, the player + timedtext fetch, the pipeline path.

All network I/O is mocked with respx.
"""

import json

import httpx
import pytest
import respx

from src import failure_reason
from src.pipeline import Pipeline
from src.providers import _url_guard
from src.providers.base import ProviderError
from src.providers.youtube import PLAYER_ENDPOINT, fetch_transcript, video_id
from tests.conftest import _clear_provider_env

VIDEO = "aA9R8bJAf68"
TIMEDTEXT = "https://www.youtube.com/api/timedtext"

# A real timedtext body escapes its text once more on top of the XML layer, so
# an apostrophe arrives as `&amp;#39;` and a tag as `&lt;i&gt;`.
TRANSCRIPT_XML = (
    '<?xml version="1.0" encoding="utf-8" ?><transcript>'
    '<text start="1.92" dur="6.52">It&amp;#39;s the first</text>'
    '<text start="5.12" dur="6.559">snippet &lt;i&gt;here&lt;/i&gt;</text>'
    '<text start="62.5" dur="3.0">a minute later</text>'
    '<text start="3700" dur="3.0">past the hour</text>'
    "</transcript>"
)


def _track(lang: str, *, asr: bool = False, name: str = "") -> dict:
    """One captionTracks entry the way the ANDROID player answers it."""
    kind = "&kind=asr" if asr else ""
    track = {
        "baseUrl": f"{TIMEDTEXT}?v={VIDEO}&lang={lang}{kind}&fmt=srv3",
        "languageCode": lang,
        "name": {"runs": [{"text": name or lang}]},
    }
    if asr:
        track["kind"] = "asr"
    return track


def _player(tracks: list[dict] | None, *, status: str = "OK", reason: str = "") -> dict:
    """A player API answer with the given caption tracks (None = no captions)."""
    body: dict = {
        "playabilityStatus": {"status": status},
        "videoDetails": {
            "videoId": VIDEO,
            "title": "Test Video",
            "author": "Test Channel",
            "lengthSeconds": "3725",
            "shortDescription": "About the video.",
        },
    }
    if reason:
        body["playabilityStatus"]["reason"] = reason
    if tracks is not None:
        body["captions"] = {"playerCaptionsTracklistRenderer": {"captionTracks": tracks}}
    return body


# -- video_id ----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VIDEO}",
        f"https://youtube.com/watch?v={VIDEO}&t=42s",
        f"https://m.youtube.com/watch?feature=share&v={VIDEO}",
        f"https://www.youtube.com/shorts/{VIDEO}",
        f"https://www.youtube.com/live/{VIDEO}?si=abc",
        f"https://www.youtube.com/embed/{VIDEO}",
        f"https://youtu.be/{VIDEO}",
        f"https://youtu.be/{VIDEO}?t=10",
    ],
)
def test_video_id_recognises_every_url_shape(url):
    assert video_id(url) == VIDEO


@pytest.mark.parametrize(
    "url",
    [
        f"https://example.com/watch?v={VIDEO}",  # not YouTube
        "https://www.youtube.com/watch",  # no id
        "https://www.youtube.com/@somechannel/videos",  # a channel page
        "https://www.youtube.com/watch?v=tooShort",  # not an 11-char id
        "https://youtu.be/",  # no id
        "http://[::1",  # malformed: must not raise
    ],
)
def test_video_id_is_none_for_anything_else(url):
    assert video_id(url) is None


# -- fetch_transcript ----------------------------------------------------------


@respx.mock
async def test_fetch_transcript_renders_markdown():
    player = respx.post(PLAYER_ENDPOINT).mock(
        return_value=httpx.Response(
            200, json=_player([_track("ru", asr=True, name="Russian (auto-generated)")])
        )
    )
    timedtext = respx.get(TIMEDTEXT).mock(return_value=httpx.Response(200, text=TRANSCRIPT_XML))

    async with httpx.AsyncClient() as client:
        markdown = await fetch_transcript(client, VIDEO, retries=0)

    sent = json.loads(player.calls.last.request.content)
    assert sent["videoId"] == VIDEO
    assert sent["context"]["client"]["clientName"] == "ANDROID"
    # The srv3 format is dropped from the track url; the rest of it is kept.
    params = timedtext.calls.last.request.url.params
    assert "fmt" not in params
    assert params["lang"] == "ru"

    assert markdown.startswith("# Test Video\n")
    assert (
        "Channel: Test Channel · Duration: 1:02:05 · Captions: Russian (auto-generated)" in markdown
    )
    assert "## Description\n\nAbout the video." in markdown
    # Unescaped text, tags stripped; a new paragraph once a minute has passed;
    # h:mm:ss from an hour up.
    assert "## Transcript\n\n[0:01] It's the first snippet here\n\n[1:02] a minute later" in markdown
    assert markdown.endswith("[1:01:40] past the hour")


@respx.mock
async def test_fetch_transcript_prefers_manual_track_in_the_spoken_language():
    tracks = [
        _track("de", name="German"),
        _track("en", asr=True, name="English (auto-generated)"),
        _track("en-GB", name="English (United Kingdom)"),
    ]
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=_player(tracks)))
    manual = respx.get(TIMEDTEXT, params={"lang": "en-GB"}).mock(
        return_value=httpx.Response(200, text=TRANSCRIPT_XML)
    )
    other = respx.get(TIMEDTEXT).mock(return_value=httpx.Response(200, text=TRANSCRIPT_XML))

    async with httpx.AsyncClient() as client:
        markdown = await fetch_transcript(client, VIDEO, retries=0)

    assert manual.call_count == 1
    assert other.call_count == 0
    assert "Captions: English (United Kingdom)" in markdown


@respx.mock
async def test_fetch_transcript_without_asr_takes_the_default_caption_track():
    # No auto-generated track: the list is alphabetical, so the first entry is
    # just a language name; YouTube's own default (the audio track's
    # defaultCaptionTrackIndex) names the one to read.
    body = _player([_track("af", name="Afrikaans"), _track("en", name="English")])
    renderer = body["captions"]["playerCaptionsTracklistRenderer"]
    renderer["audioTracks"] = [{"defaultCaptionTrackIndex": 1}]
    renderer["defaultAudioTrackIndex"] = 0
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=body))
    english = respx.get(TIMEDTEXT, params={"lang": "en"}).mock(
        return_value=httpx.Response(200, text=TRANSCRIPT_XML)
    )

    async with httpx.AsyncClient() as client:
        markdown = await fetch_transcript(client, VIDEO, retries=0)

    assert english.call_count == 1
    assert "Captions: English" in markdown


@pytest.mark.parametrize("audio_tracks", [None, [{"defaultCaptionTrackIndex": 5}]])
@respx.mock
async def test_fetch_transcript_without_asr_or_usable_default_takes_the_first_track(
    audio_tracks,
):
    body = _player([_track("af", name="Afrikaans"), _track("en", name="English")])
    if audio_tracks is not None:
        body["captions"]["playerCaptionsTracklistRenderer"]["audioTracks"] = audio_tracks
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=body))
    first = respx.get(TIMEDTEXT, params={"lang": "af"}).mock(
        return_value=httpx.Response(200, text=TRANSCRIPT_XML)
    )

    async with httpx.AsyncClient() as client:
        await fetch_transcript(client, VIDEO, retries=0)

    assert first.call_count == 1


@respx.mock
async def test_fetch_transcript_without_captions_is_classified_empty():
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=_player(None)))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, VIDEO, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


@respx.mock
async def test_fetch_transcript_bot_wall_is_classified_bot_protection():
    body = _player(
        None, status="LOGIN_REQUIRED", reason="Sign in to confirm you’re not a bot"
    )
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=body))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, VIDEO, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.BOT_PROTECTION


# -- Pipeline.read -------------------------------------------------------------


def _public_dns(monkeypatch) -> None:
    """Resolve every host to a public address, so the SSRF guard needs no DNS."""

    async def _fake_resolve(host: str) -> list[str]:
        return ["142.250.74.46"]

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)


@respx.mock
async def test_read_youtube_url_returns_the_transcript(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    url = f"https://www.youtube.com/watch?v={VIDEO}"
    respx.post(PLAYER_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_player([_track("ru", asr=True)]))
    )
    respx.get(TIMEDTEXT).mock(return_value=httpx.Response(200, text=TRANSCRIPT_XML))
    probe = respx.get(url).mock(return_value=httpx.Response(200, text="<html></html>"))
    jina = respx.get(f"https://r.jina.ai/{url}").mock(return_value=httpx.Response(500))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(url)
    finally:
        await pipe.aclose()

    assert outcome.provider == "youtube"
    assert outcome.tried == ["youtube"]
    assert outcome.failures == []
    assert "[1:02] a minute later" in outcome.markdown
    # The probe and the read chain never ran.
    assert probe.call_count == 0
    assert jina.call_count == 0


@respx.mock
async def test_read_youtube_routes_via_youtube_proxy(monkeypatch, settings):
    # Prod cannot reach youtube.com directly, so YOUTUBE_PROXY must reach the
    # client the transcript is fetched with. respx intercepts above the SOCKS
    # transport, like in test_proxied_provider_still_serves.
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    proxy = "socks5://proxy.invalid:1080"
    monkeypatch.setenv("YOUTUBE_PROXY", proxy)
    respx.post(PLAYER_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_player([_track("ru", asr=True)]))
    )
    respx.get(TIMEDTEXT).mock(return_value=httpx.Response(200, text=TRANSCRIPT_XML))

    pipe = Pipeline.build(settings)
    requested: list[str | None] = []
    client_for = pipe._clients.client_for

    def _spy(wanted: str | None) -> httpx.AsyncClient:
        requested.append(wanted)
        return client_for(wanted)

    monkeypatch.setattr(pipe._clients, "client_for", _spy)
    try:
        outcome = await pipe.read(f"https://www.youtube.com/watch?v={VIDEO}")
    finally:
        await pipe.aclose()

    assert outcome.provider == "youtube"
    assert requested == [proxy]


@respx.mock
async def test_read_youtube_video_without_captions_falls_through(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    url = f"https://www.youtube.com/watch?v={VIDEO}"
    respx.post(PLAYER_ENDPOINT).mock(return_value=httpx.Response(200, json=_player(None)))
    # The watch page itself is only chrome: trafilatura finds nothing, jina wins.
    respx.get(url).mock(
        return_value=httpx.Response(200, text="<html><body><div id='app'></div></body></html>")
    )
    jina_md = "# Test Video\n\n" + ("Video page content. " * 40)
    respx.get(f"https://r.jina.ai/{url}").mock(return_value=httpx.Response(200, text=jina_md))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(url)
    finally:
        await pipe.aclose()

    assert outcome.provider == "jina"
    assert outcome.tried == ["youtube", "trafilatura", "jina"]
    assert outcome.failures[0] == ("youtube", "empty")
