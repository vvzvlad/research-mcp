"""Instagram via its logged-out GraphQL: reel transcripts (audio via Groq) and profile post lists.

Two read providers, both ``UrlSpecificReader``s the read pipeline offers the
urls they accept before the probe. ``instagram`` (``InstagramRead``, needs the
Groq key) accepts the urls ``shortcode`` recognises as an Instagram post and
answers with ``fetch_transcript``. A plain fetch of a post page yields a login
wall; the speech lives behind two calls:

1. ``POST /api/graphql`` anonymously (no cookies, no home-page fetch), the
   logged-out post query → the author, the caption and the video's DASH
   manifest;
2. ``POST`` to Groq's Whisper endpoint with the audio url — Groq fetches the
   audio itself, it is never downloaded here.

The result is rendered as Markdown: author, duration, language, the caption and
the transcript cut into timestamped paragraphs.

``instagram_profile`` (``InstagramProfileRead``, no key) accepts the urls
``profile`` recognises as a profile and answers with ``fetch_profile_posts``:
one page of the profile's posts through the same GraphQL endpoint (the
logged-out profile posts query) — date, kind, url and caption per post, plus
the url of the next page. No Groq call is involved.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import secrets
from urllib.parse import parse_qs, quote, urlsplit
import xml.etree.ElementTree as ET

import httpx

from src import failure_reason
from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError
from src.providers.registry import register
from src.providers.youtube import _clock, _paragraphs

GRAPHQL_ENDPOINT = "https://www.instagram.com/api/graphql"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

# The logged-out post query of instagram.com's own web client.
_APP_ID = "936619743392459"
_FRIENDLY_NAME = "PolarisLoggedOutDesktopWWWPostRootContentQuery"
_DOC_ID = "27130156389949648"
# The logged-out profile posts tab query: one page of a profile's posts.
_PROFILE_FRIENDLY_NAME = "PolarisLoggedOutDesktopWWWProfilePostsTabContentQuery"
_PROFILE_DOC_ID = "27553725110923321"
_PROFILE_PAGE_SIZE = 12

# A media pk is a snowflake id: milliseconds since this epoch, shifted left 23 bits.
_PK_EPOCH_MS = 1314220021721

_WHISPER_MODEL = "whisper-large-v3-turbo"

# Provider names for request_with_retry's error messages: one per upstream, so
# a failure says which of the two calls broke.
_PROVIDER = "instagram"
_GROQ_PROVIDER = "groq"

# A shortcode is the media id written in url-safe base64 with this alphabet.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_SHORTCODE = re.compile(r"[A-Za-z0-9_-]+")
_USERNAME = re.compile(r"[A-Za-z0-9._]{1,30}")

_INSTAGRAM_HOSTS = frozenset({"instagram.com", "www.instagram.com", "m.instagram.com"})
# /p/ID, /reel/ID, /reels/ID, /tv/ID ...
_POST_PREFIXES = frozenset({"p", "reel", "reels", "tv"})
# ... and /<username>/reel/ID, /<username>/p/ID.
_USER_POST_PREFIXES = frozenset({"p", "reel"})

_MPD_NS = {"mpd": "urn:mpeg:dash:schema:mpd:2011"}


def shortcode(url: str) -> str | None:
    """The shortcode of an Instagram post url, or ``None``.

    Recognises ``instagram.com`` / ``www.`` / ``m.`` with ``/p/ID``,
    ``/reel/ID``, ``/reels/ID``, ``/tv/ID``, ``/<username>/reel/ID`` and
    ``/<username>/p/ID``; the query string is ignored. Anything else — another
    host, a profile, a story, a malformed url — is ``None``; this never raises.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if host not in _INSTAGRAM_HOSTS:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) == 2 and segments[0] in _POST_PREFIXES:
        candidate = segments[1]
    elif len(segments) == 3 and segments[1] in _USER_POST_PREFIXES:
        candidate = segments[2]
    else:
        return None
    return candidate if _SHORTCODE.fullmatch(candidate) else None


def profile(url: str) -> tuple[str, str | None] | None:
    """The ``(username, cursor)`` of an Instagram profile url, or ``None``.

    Recognises the same hosts as ``shortcode`` with a path of exactly one
    segment that is a valid username (``/<username>`` or ``/<username>/``);
    ``cursor`` is the ``after`` query parameter, ``None`` when absent. Anything
    else is ``None``; this never raises.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if host not in _INSTAGRAM_HOSTS:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) != 1 or not _USERNAME.fullmatch(segments[0]):
        return None
    after = parse_qs(parts.query).get("after")
    return segments[0], (after[0] if after else None)


async def _graphql(
    client: httpx.AsyncClient,
    friendly_name: str,
    doc_id: str,
    variables: dict,
    retries: int,
) -> object:
    """POST one logged-out query to Instagram's web GraphQL; the parsed JSON.

    Raises ``ProviderError`` when Instagram answers with a page instead of JSON
    (worded as bot protection).
    """
    lsd = secrets.token_urlsafe(8)
    response = await request_with_retry(
        client,
        "POST",
        GRAPHQL_ENDPOINT,
        retries=retries,
        provider=_PROVIDER,
        data={
            "lsd": lsd,
            "fb_api_caller_class": "RelayModern",
            "fb_api_req_friendly_name": friendly_name,
            "server_timestamps": "true",
            "variables": json.dumps(variables, separators=(",", ":")),
            "doc_id": doc_id,
        },
        # Measured: without the three Sec-Fetch-* headers Instagram answers 200
        # with an HTML page instead of JSON.
        headers={
            "X-IG-App-ID": _APP_ID,
            "X-FB-LSD": lsd,
            "X-FB-Friendly-Name": friendly_name,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        },
    )
    try:
        return response.json()
    except ValueError as exc:
        raise ProviderError(
            "instagram: blocked by bot protection (an HTML page instead of JSON)",
            reason=failure_reason.BOT_PROTECTION,
        ) from exc


def _media_id(code: str) -> str:
    """The numeric media id (``pk``) a shortcode encodes."""
    pk = 0
    for char in code:
        pk = pk * 64 + _ALPHABET.index(char)
    return str(pk)


def _audio_url(media: dict) -> str | None:
    """The url Groq should transcribe: the leanest DASH audio track, else the video.

    ``None`` when the post carries neither a DASH audio track nor a video file.
    """
    manifest = media.get("video_dash_manifest")
    if manifest:
        root = ET.fromstring(manifest)
        audio = root.find(".//mpd:AdaptationSet[@contentType='audio']", _MPD_NS)
        representations = audio.findall("mpd:Representation", _MPD_NS) if audio is not None else []
        if representations:
            leanest = min(representations, key=lambda rep: int(rep.get("bandwidth", "0")))
            base_url = leanest.findtext("mpd:BaseURL", namespaces=_MPD_NS)
            if base_url:
                return base_url
    versions = media.get("video_versions") or []
    return versions[0].get("url") if versions else None


async def fetch_transcript(
    instagram_client: httpx.AsyncClient,
    groq_client: httpx.AsyncClient,
    shortcode: str,
    api_key: str,
    retries: int,
) -> str:
    """Fetch the post ``shortcode``, transcribe its audio and render Markdown.

    Raises ``ProviderError`` when Instagram answers with a page instead of JSON
    (worded as bot protection), when the post is not visible logged out or has
    no video (worded as an empty response), or when Groq's answer cannot be
    parsed.
    """
    payload = await _graphql(
        instagram_client, _FRIENDLY_NAME, _DOC_ID, {"media_id": _media_id(shortcode)}, retries
    )
    data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
    media = (data.get("xig_polaris_media") or {}).get("if_not_gated_logged_out")
    if not media:
        raise ProviderError(
            "instagram: empty response (private, deleted or login-gated post)",
            reason=failure_reason.EMPTY,
        )
    audio_url = _audio_url(media)
    if not audio_url:
        raise ProviderError(
            "instagram: empty response (the post has no video)", reason=failure_reason.EMPTY
        )

    # Groq rejects a form-urlencoded body: the fields must go as multipart.
    transcribed = await request_with_retry(
        groq_client,
        "POST",
        GROQ_ENDPOINT,
        retries=retries,
        provider=_GROQ_PROVIDER,
        headers={"Authorization": f"Bearer {api_key}"},
        files={
            "model": (None, _WHISPER_MODEL),
            "url": (None, audio_url),
            "response_format": (None, "verbose_json"),
        },
    )
    try:
        transcription = transcribed.json()
    except ValueError as exc:
        raise ProviderError(f"groq: invalid transcription response: {exc}") from exc
    if not isinstance(transcription, dict):
        raise ProviderError("groq: invalid transcription response: not a JSON object")

    snippets = [
        (float(segment.get("start") or 0), (segment.get("text") or "").strip())
        for segment in transcription.get("segments") or []
    ]
    snippets = [(start, text) for start, text in snippets if text]

    user = media.get("user") or {}
    duration = _clock(float(transcription.get("duration") or 0))
    lines = [
        f"# @{user.get('username', '')} on Instagram",
        "",
        f"Author: {user.get('full_name', '')} · Duration: {duration} · "
        f"Language: {transcription.get('language', '')}",
        "",
    ]
    caption = ((media.get("caption") or {}).get("text") or "").strip()
    if caption:
        lines += ["## Caption", "", caption, ""]
    transcript = "\n\n".join(_paragraphs(snippets)) if snippets else "_No speech in the audio._"
    lines += ["## Transcript", "", transcript]
    return "\n".join(lines)


def _post_kind(node: dict) -> str:
    """What a timeline node is: ``reel``, ``carousel``, ``video`` or ``photo``."""
    product_type = node.get("product_type")
    if product_type == "clips":
        return "reel"
    if product_type == "carousel_container":
        return "carousel"
    if node.get("__typename") == "XIGPolarisVideoMedia":
        return "video"
    return "photo"


def _post_date(pk: str) -> str:
    """The ``YYYY-MM-DD`` (UTC) a media pk encodes."""
    seconds = ((int(pk) >> 23) + _PK_EPOCH_MS) // 1000
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%d")


async def fetch_profile_posts(
    client: httpx.AsyncClient,
    username: str,
    cursor: str | None,
    retries: int,
) -> str:
    """Fetch one page of ``username``'s posts (after ``cursor``) and render Markdown.

    Raises ``ProviderError`` when Instagram answers with a page instead of JSON
    (worded as bot protection), or when there is no such public profile or it
    shows no posts (worded as an empty response).
    """
    variables: dict = {"first": _PROFILE_PAGE_SIZE, "username": username}
    if cursor:
        variables["after"] = cursor
    payload = await _graphql(client, _PROFILE_FRIENDLY_NAME, _PROFILE_DOC_ID, variables, retries)
    data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
    user = data.get("xig_user_by_username")
    if not user:
        raise ProviderError(
            "instagram: empty response (no such public profile)", reason=failure_reason.EMPTY
        )
    timeline = user.get("polaris_ordered_timeline_connection") or {}
    edges = timeline.get("edges") or []
    if not edges:
        raise ProviderError(
            "instagram: empty response (the profile shows no posts)",
            reason=failure_reason.EMPTY,
        )

    lines = [f"# @{username} on Instagram — posts"]
    for edge in edges:
        node = edge.get("node") or {}
        kind = _post_kind(node)
        section = "reel" if kind == "reel" else "p"
        url = f"https://www.instagram.com/{section}/{node.get('code', '')}/"
        lines += ["", f"## {_post_date(node.get('pk', '0'))} · {kind} · {url}"]
        caption = ((node.get("caption") or {}).get("text") or "").strip()
        if caption:
            lines += ["", caption]

    page_info = timeline.get("page_info") or {}
    end_cursor = page_info.get("end_cursor")
    if page_info.get("has_next_page") and end_cursor:
        next_url = f"https://www.instagram.com/{username}/?after={quote(end_cursor, safe='')}"
        lines += ["", f"Next page: {next_url}"]
    else:
        lines += ["", "No more posts."]
    return "\n".join(lines)


@register("instagram")
class InstagramRead:
    """Read an Instagram post url as a transcript of its audio (requires the Groq ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise ValueError("instagram requires an api_key (the Groq key)")
        self.name = config.name
        self.proxy = config.proxy
        self._config = config
        self._groq_proxy = config.options.get("groq_proxy")
        self._groq_client: httpx.AsyncClient | None = None

    def _groq(self) -> httpx.AsyncClient:
        # The pipeline hands a reader ONE client, bound to its `proxy` (the
        # Instagram one), but Groq is a second upstream with a proxy of its own,
        # so this reader owns that client. Created lazily: the reader is built
        # before the event loop starts, and a client created there would be
        # bound to the wrong loop. Re-created if closed, so a closed client is
        # never reused.
        if self._groq_client is None or self._groq_client.is_closed:
            self._groq_client = httpx.AsyncClient(
                timeout=self._config.request_timeout, proxy=self._groq_proxy
            )
        return self._groq_client

    def accepts(self, url: str) -> bool:
        return shortcode(url) is not None

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        return await fetch_transcript(
            client, self._groq(), shortcode(url), self._config.api_key, self._config.retries
        )


@register("instagram_profile")
class InstagramProfileRead:
    """Read an Instagram profile url as one page of its posts (no key needed)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.name = config.name
        self.proxy = config.proxy
        self._config = config

    def accepts(self, url: str) -> bool:
        return profile(url) is not None

    async def read(self, client: httpx.AsyncClient, url: str) -> str:
        username, cursor = profile(url)
        return await fetch_profile_posts(client, username, cursor, self._config.retries)
