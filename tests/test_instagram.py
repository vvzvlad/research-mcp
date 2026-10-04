"""Instagram reel transcripts and profile posts: url detection, the fetches, the pipeline paths.

All network I/O is mocked with respx.
"""

import json
import re
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from src import failure_reason
from src.pipeline import Pipeline
from src.providers import _url_guard
from src.providers.base import ProviderError
from src.providers.instagram import (
    GRAPHQL_ENDPOINT,
    GROQ_ENDPOINT,
    fetch_profile_posts,
    fetch_transcript,
    profile,
    shortcode,
)
from tests.conftest import _clear_provider_env, _jina_answer

SHORTCODE = "DbIv3T_xMUn"
# The shortcode read as url-safe base64 (A-Za-z0-9-_), i.e.
# int.from_bytes(base64.urlsafe_b64decode("A" + SHORTCODE), "big").
MEDIA_ID = "3947615582618436903"
POST_URL = f"https://www.instagram.com/reel/{SHORTCODE}/"
GROQ_KEY = "gsk-test"

VIDEO_URL = "https://cdn.example/video.mp4?x=1"
LEAN_AUDIO_URL = "https://cdn.example/audio-lo.mp4?a=1&b=2"

# Video listed first with the lowest bandwidth of all, the lean audio track
# listed last: only the audio set's own minimum may be picked.
MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static"><Period>'
    '<AdaptationSet contentType="video">'
    '<Representation id="v1" bandwidth="30000"><BaseURL>https://cdn.example/v.mp4</BaseURL>'
    "</Representation></AdaptationSet>"
    '<AdaptationSet contentType="audio">'
    '<Representation id="a1" bandwidth="128000">'
    "<BaseURL>https://cdn.example/audio-hi.mp4?a=1&amp;b=2</BaseURL></Representation>"
    '<Representation id="a2" bandwidth="64000">'
    "<BaseURL>https://cdn.example/audio-lo.mp4?a=1&amp;b=2</BaseURL></Representation>"
    "</AdaptationSet></Period></MPD>"
)
MANIFEST_WITHOUT_AUDIO = (
    '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"><Period>'
    '<AdaptationSet contentType="video">'
    '<Representation id="v1" bandwidth="30000"><BaseURL>https://cdn.example/v.mp4</BaseURL>'
    "</Representation></AdaptationSet></Period></MPD>"
)

GROQ_ANSWER = {
    "text": "Привет всем. Это первый ролик. Минута прошла.",
    "language": "Russian",
    "duration": 75.4,
    "segments": [
        {"start": 0.0, "text": " Привет всем."},
        {"start": 4.2, "text": " Это первый ролик."},
        {"start": 12.0, "text": "  "},
        {"start": 65.0, "text": " Минута прошла."},
    ],
}

EXPECTED_MARKDOWN = (
    "# @testuser on Instagram\n"
    "\n"
    "Author: Test User · Duration: 1:15 · Language: Russian\n"
    "\n"
    "## Caption\n"
    "\n"
    "About the reel.\n"
    "\n"
    "## Transcript\n"
    "\n"
    "[0:00] Привет всем. Это первый ролик.\n"
    "\n"
    "[1:05] Минута прошла."
)


def _post(
    *, manifest: str | None = MANIFEST, versions: list | None = None, caption="About the reel."
) -> dict:
    """A GraphQL answer for a post visible logged out."""
    media: dict = {
        "user": {"username": "testuser", "full_name": "Test User"},
        "caption": {"text": caption} if caption is not None else None,
        "video_dash_manifest": manifest,
        "video_versions": versions if versions is not None else [{"url": VIDEO_URL}],
    }
    return {"data": {"xig_polaris_media": {"if_not_gated_logged_out": media}}}


GATED = {"data": {"xig_polaris_media": {"if_not_gated_logged_out": None}}}


def _multipart_fields(request: httpx.Request) -> dict[str, str]:
    """The fields of a multipart/form-data request body (fails on any other body)."""
    content_type = request.headers["Content-Type"]
    assert content_type.startswith("multipart/form-data")
    boundary = content_type.split("boundary=")[1]
    fields: dict[str, str] = {}
    for part in request.content.decode().split(f"--{boundary}"):
        head, separator, value = part.partition("\r\n\r\n")
        name = re.search(r'name="([^"]+)"', head)
        if separator and name:
            fields[name.group(1)] = value.removesuffix("\r\n")
    return fields


# -- shortcode -----------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.instagram.com/reel/{SHORTCODE}/?stkn=MTA1OTJudTRndWlsNw==",
        f"https://www.instagram.com/reel/{SHORTCODE}",
        f"https://instagram.com/p/{SHORTCODE}/",
        f"https://m.instagram.com/reels/{SHORTCODE}/",
        f"https://www.instagram.com/tv/{SHORTCODE}/",
        f"https://www.instagram.com/someuser/reel/{SHORTCODE}/",
        f"https://www.instagram.com/someuser/p/{SHORTCODE}/?img_index=1",
    ],
)
def test_shortcode_recognises_every_url_shape(url):
    assert shortcode(url) == SHORTCODE


@pytest.mark.parametrize(
    "url",
    [
        f"https://example.com/reel/{SHORTCODE}/",  # not Instagram
        "https://www.instagram.com/someuser/",  # a profile
        "https://www.instagram.com/someuser/reels/",  # a profile's reels tab
        "https://www.instagram.com/stories/someuser/3456789012345678901/",  # a story
        "https://www.instagram.com/reels/audio/123456789/",  # an audio page
        "https://www.instagram.com/reel/",  # no id
        "https://www.instagram.com/reel/bad!id/",  # not a shortcode
        "http://[::1",  # malformed: must not raise
    ],
)
def test_shortcode_is_none_for_anything_else(url):
    assert shortcode(url) is None


# -- fetch_transcript ------------------------------------------------------------


@respx.mock
async def test_fetch_transcript_renders_markdown():
    graphql = respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_post()))
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    async with httpx.AsyncClient() as client:
        markdown = await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    request = graphql.calls.last.request
    form = {key: values[0] for key, values in parse_qs(request.content.decode()).items()}
    assert form["variables"] == json.dumps({"media_id": MEDIA_ID}, separators=(",", ":"))
    assert form["doc_id"] == "27130156389949648"
    assert form["fb_api_req_friendly_name"] == "PolarisLoggedOutDesktopWWWPostRootContentQuery"
    assert form["lsd"] == request.headers["X-FB-LSD"]
    assert request.headers["X-IG-App-ID"] == "936619743392459"
    assert request.headers["Sec-Fetch-Site"] == "same-origin"
    assert request.headers["Sec-Fetch-Mode"] == "cors"
    assert request.headers["Sec-Fetch-Dest"] == "empty"

    sent = groq.calls.last.request
    assert sent.headers["Authorization"] == f"Bearer {GROQ_KEY}"
    assert _multipart_fields(sent) == {
        "model": "whisper-large-v3-turbo",
        "url": LEAN_AUDIO_URL,
        "response_format": "verbose_json",
    }

    assert markdown == EXPECTED_MARKDOWN


@pytest.mark.parametrize("manifest", [None, MANIFEST_WITHOUT_AUDIO])
@respx.mock
async def test_fetch_transcript_without_dash_audio_sends_the_video_url(manifest):
    respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_post(manifest=manifest))
    )
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    async with httpx.AsyncClient() as client:
        await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert _multipart_fields(groq.calls.last.request)["url"] == VIDEO_URL


@respx.mock
async def test_fetch_transcript_without_speech_or_caption():
    respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_post(caption=None))
    )
    silent = {
        "text": "",
        "language": "English",
        "duration": 9.0,
        "segments": [{"start": 0.0, "text": " "}],
    }
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=silent))

    async with httpx.AsyncClient() as client:
        markdown = await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert "## Caption" not in markdown
    assert markdown.endswith("## Transcript\n\n_No speech in the audio._")


@respx.mock
async def test_fetch_transcript_html_answer_is_classified_bot_protection():
    respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(
            200, text="<!DOCTYPE html><html><body>Login</body></html>",
            headers={"Content-Type": "text/html"},
        )
    )
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.BOT_PROTECTION
    assert groq.call_count == 0


@respx.mock
async def test_fetch_transcript_gated_post_is_classified_empty():
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=GATED))
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.EMPTY
    assert groq.call_count == 0


@respx.mock
async def test_fetch_transcript_post_without_video_is_classified_empty():
    respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_post(manifest=None, versions=[]))
    )
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert "no video" in str(caught.value)
    assert failure_reason.classify(caught.value) == failure_reason.EMPTY
    assert groq.call_count == 0


@respx.mock
async def test_fetch_transcript_unparseable_groq_answer_is_a_provider_error():
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_post()))
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, text="not json"))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_transcript(client, client, SHORTCODE, GROQ_KEY, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.OTHER


# -- Pipeline.read -------------------------------------------------------------


def _public_dns(monkeypatch) -> None:
    """Resolve every host to a public address, so the SSRF guard needs no DNS."""

    async def _fake_resolve(host: str) -> list[str]:
        return ["157.240.0.174"]

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)


@respx.mock
async def test_read_instagram_url_returns_the_transcript(monkeypatch, settings, capture_logs):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_post()))
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))
    probe = respx.get(POST_URL).mock(return_value=httpx.Response(200, text="<html></html>"))
    jina = respx.get(f"https://r.jina.ai/{POST_URL}").mock(return_value=httpx.Response(500))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(POST_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "instagram"
    assert outcome.tried == ["instagram"]
    assert outcome.failures == []
    assert outcome.markdown == EXPECTED_MARKDOWN
    # The probe and the read chain never ran.
    assert probe.call_count == 0
    assert jina.call_count == 0
    # Groq bills the transcription, so the success counts as a paid call.
    line = next(m for m in capture_logs if "provider=instagram ok=true" in m)
    assert "paid_calls=1" in line


@respx.mock
async def test_read_instagram_routes_via_instagram_and_groq_proxies(monkeypatch, settings):
    # Each upstream gets the client bound to its own proxy. respx intercepts
    # above the SOCKS transport, like in test_read_youtube_routes_via_youtube_proxy.
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    instagram_proxy = "socks5://instagram-proxy.invalid:1080"
    groq_proxy = "socks5://groq-proxy.invalid:1080"
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    monkeypatch.setenv("INSTAGRAM_PROXY", instagram_proxy)
    monkeypatch.setenv("GROQ_PROXY", groq_proxy)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_post()))
    respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))

    pipe = Pipeline.build(settings)
    requested: list[str | None] = []
    client_for = pipe._clients.client_for

    def _spy(wanted: str | None) -> httpx.AsyncClient:
        requested.append(wanted)
        return client_for(wanted)

    monkeypatch.setattr(pipe._clients, "client_for", _spy)
    try:
        outcome = await pipe.read(POST_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "instagram"
    assert requested == [instagram_proxy, groq_proxy]


@respx.mock
async def test_read_instagram_failure_falls_through(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=GATED))
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))
    # The post page itself is a login wall: trafilatura finds nothing, jina wins.
    respx.get(POST_URL).mock(
        return_value=httpx.Response(200, text="<html><body><div id='app'></div></body></html>")
    )
    jina_md = "# Instagram\n\n" + ("Reel page content. " * 40)
    respx.get(f"https://r.jina.ai/{POST_URL}").mock(return_value=_jina_answer(jina_md))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(POST_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "jina"
    assert outcome.tried == ["instagram", "trafilatura", "jina"]
    assert outcome.failures[0] == ("instagram", "empty")
    assert groq.call_count == 0


@respx.mock
async def test_read_instagram_without_groq_key_skips_the_path(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    graphql = respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_post()))
    respx.get(POST_URL).mock(
        return_value=httpx.Response(200, text="<html><body><div id='app'></div></body></html>")
    )
    jina_md = "# Instagram\n\n" + ("Reel page content. " * 40)
    respx.get(f"https://r.jina.ai/{POST_URL}").mock(return_value=_jina_answer(jina_md))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(POST_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "jina"
    assert outcome.tried == ["trafilatura", "jina"]
    assert graphql.call_count == 0


# -- profile posts ---------------------------------------------------------------

PROFILE_URL = "https://www.instagram.com/testuser/"
CURSOR = "QVFB+abc/def=="
NEXT_CURSOR = "QVFC+next/page=="

# Each pk is ((ms - 1314220021721) << 23) for noon-ish UTC on the commented day.
PROFILE_NODES = [
    {  # pinned: older than the posts after it, the order must be kept as is
        "code": "PHOTO123",
        "pk": "3799663432152645632",  # 2025-12-31
        "__typename": "XIGPolarisImageMedia",
        "product_type": "feed",
        "caption": {"text": ""},
        "user": {"username": "testuser"},
    },
    {
        "code": "REEL123",
        "pk": "3997587604747845632",  # 2026-09-30
        "__typename": "XIGPolarisVideoMedia",
        "product_type": "clips",
        "caption": {"text": "About the reel."},
        "user": {"username": "testuser"},
    },
    {
        "code": "CAROUSEL1",
        "pk": "3964142224651845632",  # 2026-08-15
        "__typename": "XIGPolarisCarouselMedia",
        "product_type": "carousel_container",
        "caption": {"text": "  Three photos.\n"},
        "user": {"username": "testuser"},
    },
    {
        "code": "VIDEO123",
        "pk": "3931965202085445632",  # 2026-07-01
        "__typename": "XIGPolarisVideoMedia",
        "product_type": "feed",
        "caption": None,
        "user": {"username": "testuser"},
    },
]

EXPECTED_PROFILE_MARKDOWN = (
    "# @testuser on Instagram — posts\n"
    "\n"
    "## 2025-12-31 · photo · https://www.instagram.com/p/PHOTO123/\n"
    "\n"
    "## 2026-09-30 · reel · https://www.instagram.com/reel/REEL123/\n"
    "\n"
    "About the reel.\n"
    "\n"
    "## 2026-08-15 · carousel · https://www.instagram.com/p/CAROUSEL1/\n"
    "\n"
    "Three photos.\n"
    "\n"
    "## 2026-07-01 · video · https://www.instagram.com/p/VIDEO123/\n"
    "\n"
    "Next page: https://www.instagram.com/testuser/?after=QVFC%2Bnext%2Fpage%3D%3D"
)


def _profile_page(
    nodes: list[dict] = PROFILE_NODES,
    *,
    end_cursor: str | None = NEXT_CURSOR,
    has_next_page: bool = True,
) -> dict:
    """A GraphQL answer for one page of a public profile's posts."""
    return {
        "data": {
            "xig_user_by_username": {
                "polaris_ordered_timeline_connection": {
                    "edges": [{"node": node} for node in nodes],
                    "page_info": {"end_cursor": end_cursor, "has_next_page": has_next_page},
                }
            }
        }
    }


UNKNOWN_PROFILE = {"data": {"xig_user_by_username": None}}


def _form(request: httpx.Request) -> dict[str, str]:
    """The fields of a form-urlencoded request body."""
    return {key: values[0] for key, values in parse_qs(request.content.decode()).items()}


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.instagram.com/tvikas.m/", ("tvikas.m", None)),
        ("https://www.instagram.com/tvikas.m", ("tvikas.m", None)),
        ("https://instagram.com/some_user/", ("some_user", None)),
        ("https://m.instagram.com/some_user", ("some_user", None)),
        ("https://www.instagram.com/some_user/?hl=en", ("some_user", None)),
        (
            "https://www.instagram.com/tvikas.m/?after=QVFB%2Babc%2Fdef%3D%3D",
            ("tvikas.m", CURSOR),
        ),
    ],
)
def test_profile_recognises_every_url_shape(url, expected):
    assert profile(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.instagram.com/reel/{SHORTCODE}/",  # a post
        f"https://www.instagram.com/p/{SHORTCODE}/",  # a post
        f"https://www.instagram.com/someuser/reel/{SHORTCODE}/",  # a post under a user
        "https://www.instagram.com/someuser/reels/",  # two segments
        "https://www.instagram.com/someuser/tagged/",  # two segments
        "https://www.instagram.com/",  # no segment
        "https://example.com/someuser/",  # not Instagram
        "https://www.instagram.com/some-user/",  # "-" is not a username character
        "https://www.instagram.com/bad!name/",  # nor is "!"
        "https://www.instagram.com/" + "a" * 31 + "/",  # longer than 30
        "http://[::1",  # malformed: must not raise
    ],
)
def test_profile_is_none_for_anything_else(url):
    assert profile(url) is None


@respx.mock
async def test_fetch_profile_posts_renders_markdown():
    graphql = respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_profile_page())
    )

    async with httpx.AsyncClient() as client:
        markdown = await fetch_profile_posts(client, "testuser", CURSOR, retries=0)

    request = graphql.calls.last.request
    form = _form(request)
    assert form["variables"] == json.dumps(
        {"first": 12, "username": "testuser", "after": CURSOR}, separators=(",", ":")
    )
    assert form["doc_id"] == "27553725110923321"
    friendly_name = "PolarisLoggedOutDesktopWWWProfilePostsTabContentQuery"
    assert form["fb_api_req_friendly_name"] == friendly_name
    assert request.headers["X-FB-Friendly-Name"] == friendly_name
    assert form["lsd"] == request.headers["X-FB-LSD"]
    assert request.headers["X-IG-App-ID"] == "936619743392459"
    assert request.headers["Sec-Fetch-Site"] == "same-origin"
    assert request.headers["Sec-Fetch-Mode"] == "cors"
    assert request.headers["Sec-Fetch-Dest"] == "empty"

    assert markdown == EXPECTED_PROFILE_MARKDOWN


@pytest.mark.parametrize(
    "page_info",
    [
        {"end_cursor": NEXT_CURSOR, "has_next_page": False},
        {"end_cursor": None, "has_next_page": True},
    ],
)
@respx.mock
async def test_fetch_profile_posts_last_page_says_no_more_posts(page_info):
    graphql = respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_profile_page(PROFILE_NODES[1:2], **page_info))
    )

    async with httpx.AsyncClient() as client:
        markdown = await fetch_profile_posts(client, "testuser", None, retries=0)

    # The first page carries no cursor at all.
    assert json.loads(_form(graphql.calls.last.request)["variables"]) == {
        "first": 12,
        "username": "testuser",
    }
    assert markdown.endswith("About the reel.\n\nNo more posts.")
    assert "Next page" not in markdown


@pytest.mark.parametrize(
    ("answer", "wording"),
    [
        (UNKNOWN_PROFILE, "no such public profile"),
        (_profile_page([]), "the profile shows no posts"),
    ],
)
@respx.mock
async def test_fetch_profile_posts_without_posts_is_classified_empty(answer, wording):
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=answer))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_profile_posts(client, "testuser", None, retries=0)

    assert wording in str(caught.value)
    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


@respx.mock
async def test_fetch_profile_posts_html_answer_is_classified_bot_protection():
    respx.post(GRAPHQL_ENDPOINT).mock(
        return_value=httpx.Response(
            200, text="<!DOCTYPE html><html><body>Login</body></html>",
            headers={"Content-Type": "text/html"},
        )
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await fetch_profile_posts(client, "testuser", None, retries=0)

    assert failure_reason.classify(caught.value) == failure_reason.BOT_PROTECTION


@pytest.mark.parametrize("groq_key", [None, GROQ_KEY])
@respx.mock
async def test_read_profile_url_returns_the_posts(monkeypatch, settings, capture_logs, groq_key):
    # The path needs no Groq key, and is never billed — not even when the key
    # puts "instagram" into the paid set for the transcript path.
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    if groq_key:
        monkeypatch.setenv("GROQ_API_KEY", groq_key)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_profile_page()))
    groq = respx.post(GROQ_ENDPOINT).mock(return_value=httpx.Response(200, json=GROQ_ANSWER))
    probe = respx.get(PROFILE_URL).mock(return_value=httpx.Response(200, text="<html></html>"))
    jina = respx.get(f"https://r.jina.ai/{PROFILE_URL}").mock(return_value=httpx.Response(500))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(PROFILE_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "instagram"
    assert outcome.tried == ["instagram"]
    assert outcome.failures == []
    assert outcome.markdown == EXPECTED_PROFILE_MARKDOWN
    # The probe, the read chain and Groq never ran.
    assert probe.call_count == 0
    assert jina.call_count == 0
    assert groq.call_count == 0
    line = next(m for m in capture_logs if "provider=instagram ok=true" in m)
    assert "paid_calls=0" in line


@respx.mock
async def test_read_profile_routes_via_instagram_proxy(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    instagram_proxy = "socks5://instagram-proxy.invalid:1080"
    monkeypatch.setenv("INSTAGRAM_PROXY", instagram_proxy)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=_profile_page()))

    pipe = Pipeline.build(settings)
    requested: list[str | None] = []
    client_for = pipe._clients.client_for

    def _spy(wanted: str | None) -> httpx.AsyncClient:
        requested.append(wanted)
        return client_for(wanted)

    monkeypatch.setattr(pipe._clients, "client_for", _spy)
    try:
        outcome = await pipe.read(PROFILE_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "instagram"
    assert requested == [instagram_proxy]


@respx.mock
async def test_read_profile_failure_falls_through(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    respx.post(GRAPHQL_ENDPOINT).mock(return_value=httpx.Response(200, json=UNKNOWN_PROFILE))
    # The profile page itself is a login wall: trafilatura finds nothing, jina wins.
    respx.get(PROFILE_URL).mock(
        return_value=httpx.Response(200, text="<html><body><div id='app'></div></body></html>")
    )
    jina_md = "# Instagram\n\n" + ("Profile page content. " * 40)
    respx.get(f"https://r.jina.ai/{PROFILE_URL}").mock(return_value=_jina_answer(jina_md))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(PROFILE_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "jina"
    assert outcome.tried == ["instagram", "trafilatura", "jina"]
    assert outcome.failures[0] == ("instagram", "empty")
