"""Instagram reel transcripts: the post via Instagram's GraphQL, the audio via Groq.

Not a registered provider — like ``youtube.py`` it is invoked directly by the
read pipeline: ``Pipeline.read`` asks ``shortcode`` whether a url is an
Instagram post and, if it is (and ``GROQ_API_KEY`` is set), tries
``fetch_transcript`` before the probe. A plain fetch of a post page yields a
login wall; the speech lives behind two calls:

1. ``POST /api/graphql`` anonymously (no cookies, no home-page fetch), the
   logged-out post query → the author, the caption and the video's DASH
   manifest;
2. ``POST`` to Groq's Whisper endpoint with the audio url — Groq fetches the
   audio itself, it is never downloaded here.

The result is rendered as Markdown: author, duration, language, the caption and
the transcript cut into timestamped paragraphs.
"""

from __future__ import annotations

import json
import re
import secrets
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderError
from src.providers.youtube import _clock, _paragraphs

GRAPHQL_ENDPOINT = "https://www.instagram.com/api/graphql"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/audio/transcriptions"

# The logged-out post query of instagram.com's own web client.
_APP_ID = "936619743392459"
_FRIENDLY_NAME = "PolarisLoggedOutDesktopWWWPostRootContentQuery"
_DOC_ID = "27130156389949648"

_WHISPER_MODEL = "whisper-large-v3-turbo"

# Provider names for request_with_retry's error messages: one per upstream, so
# a failure says which of the two calls broke.
_PROVIDER = "instagram"
_GROQ_PROVIDER = "groq"

# A shortcode is the media id written in url-safe base64 with this alphabet.
_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
_SHORTCODE = re.compile(r"[A-Za-z0-9_-]+")

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
    lsd = secrets.token_urlsafe(8)
    response = await request_with_retry(
        instagram_client,
        "POST",
        GRAPHQL_ENDPOINT,
        retries=retries,
        provider=_PROVIDER,
        data={
            "lsd": lsd,
            "fb_api_caller_class": "RelayModern",
            "fb_api_req_friendly_name": _FRIENDLY_NAME,
            "server_timestamps": "true",
            "variables": json.dumps({"media_id": _media_id(shortcode)}, separators=(",", ":")),
            "doc_id": _DOC_ID,
        },
        # Measured: without the three Sec-Fetch-* headers Instagram answers 200
        # with an HTML page instead of JSON.
        headers={
            "X-IG-App-ID": _APP_ID,
            "X-FB-LSD": lsd,
            "X-FB-Friendly-Name": _FRIENDLY_NAME,
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
        },
    )
    try:
        payload = response.json()
    except ValueError as exc:
        # src.failure_reason.classify keys on the "bot protection" wording.
        raise ProviderError(
            "instagram: blocked by bot protection (an HTML page instead of JSON)"
        ) from exc

    data = (payload.get("data") or {}) if isinstance(payload, dict) else {}
    media = (data.get("xig_polaris_media") or {}).get("if_not_gated_logged_out")
    if not media:
        # "empty response" is the wording classify() maps to `empty`.
        raise ProviderError("instagram: empty response (private, deleted or login-gated post)")
    audio_url = _audio_url(media)
    if not audio_url:
        raise ProviderError("instagram: empty response (the post has no video)")

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
