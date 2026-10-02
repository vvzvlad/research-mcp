"""YouTube video transcripts via YouTube's own player API.

Not a registered provider — like ``pdf.py`` it is invoked directly by the read
pipeline: ``Pipeline.read`` asks ``video_id`` whether a url is a YouTube video
and, if it is, tries ``fetch_transcript`` before the probe. A plain fetch of a
watch page yields only the page chrome; the transcript lives behind two calls:

1. ``POST /youtubei/v1/player`` as the ANDROID client (the one
   youtube-transcript-api 1.2.4 uses — the WEB client answers UNPLAYABLE) →
   the video's details and its caption tracks;
2. ``GET <track baseUrl>`` without ``&fmt=srv3`` → the plain timedtext XML
   (``<transcript><text start=".." dur="..">...</text>...</transcript>``).

The result is rendered as Markdown: title, channel, duration, the description
and the transcript cut into timestamped paragraphs.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderError

PLAYER_ENDPOINT = "https://www.youtube.com/youtubei/v1/player"

# The client identity sent to the player API; no `?key=` is needed with it.
_PLAYER_CLIENT = {"clientName": "ANDROID", "clientVersion": "20.10.38"}

# Provider name for request_with_retry's error messages.
_PROVIDER = "youtube"

# A YouTube video id: exactly 11 url-safe base64 characters.
_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")

# Hosts whose urls carry the id in `?v=` (/watch) or as the second path segment.
_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com"})
_ID_PATH_PREFIXES = frozenset({"shorts", "live", "embed"})

# Caption text may carry HTML tags (<font>, <i>, ...); stripped after unescaping,
# exactly like youtube-transcript-api does.
_HTML_TAG = re.compile(r"<[^>]*>")

# A transcript paragraph spans at most this many seconds of video.
_PARAGRAPH_SECONDS = 60.0


def video_id(url: str) -> str | None:
    """The 11-character video id of a YouTube video url, or ``None``.

    Recognises ``youtube.com`` / ``www.`` / ``m.`` with ``/watch?v=ID``,
    ``/shorts/ID``, ``/live/ID`` and ``/embed/ID``, plus ``youtu.be/ID``.
    Anything else — another host, a channel page, a malformed url — is ``None``;
    this never raises.
    """
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if host == "youtu.be":
        candidate = segments[0] if segments else ""
    elif host in _YOUTUBE_HOSTS:
        if segments == ["watch"]:
            candidate = parse_qs(parts.query).get("v", [""])[0]
        elif len(segments) >= 2 and segments[0] in _ID_PATH_PREFIXES:
            candidate = segments[1]
        else:
            return None
    else:
        return None
    return candidate if _VIDEO_ID.fullmatch(candidate) else None


def _clock(seconds: float) -> str:
    """``m:ss`` under an hour, ``h:mm:ss`` from an hour up."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _base_language(code: str | None) -> str:
    """``en`` for ``en``, ``en-US``, ``en-US.4`` — the comparable part of a code."""
    return (code or "").split(".")[0].split("-")[0].lower()


def _spoken_language(player: dict, tracks: list[dict]) -> str:
    """The base language actually spoken in the video ("" when unknown).

    An auto-dubbed video lists one audio track per dub in ``streamingData``, and
    the caption list then carries an ``asr`` track for EVERY dub, sorted by
    language name — so the first ``asr`` track is merely the alphabetically
    first dub (Arabic on -xZfqwRWpPg). The original audio is the one whose
    display name says "original"; the names are English because we send no
    ``hl``. Measured: ``audioIsDefault`` follows the viewer language (``hl=ru``
    moves it to the Russian dub), and the renderer's ``defaultCaptionTrackIndex``
    moves to a MANUAL track in the viewer language when one exists (TED with
    ``hl=ru`` → ru) — so neither of them names the spoken language.

    A plain video carries no audio-track info and exactly one ``asr`` track,
    which is in the spoken language.
    """
    for fmt in (player.get("streamingData") or {}).get("adaptiveFormats") or []:
        audio = fmt.get("audioTrack") or {}
        if "original" in (audio.get("displayName") or "").lower():
            return _base_language(audio.get("id"))
    asr = next((track for track in tracks if track.get("kind") == "asr"), None)
    return _base_language(asr.get("languageCode")) if asr else ""


def _pick_track(player: dict, tracks: list[dict]) -> dict:
    """Choose the caption track to read.

    A track in the spoken language wins — a manual one (written by a human,
    hence better) over the auto-generated one. When the spoken language is
    unknown or has no track (no ``asr`` track and no dub info, e.g. a video with
    only uploaded translations), YouTube's own default caption track, else the
    first one listed (alphabetical).
    """
    spoken = _spoken_language(player, tracks)
    same = [track for track in tracks if spoken and _base_language(track.get("languageCode")) == spoken]
    manual = [track for track in same if track.get("kind") != "asr"]
    if same:
        return (manual or same)[0]
    renderer = (player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
    audio = renderer.get("audioTracks") or []
    index = renderer.get("defaultAudioTrackIndex") or 0
    default = audio[index].get("defaultCaptionTrackIndex") if index < len(audio) else None
    if isinstance(default, int) and 0 <= default < len(tracks):
        return tracks[default]
    return tracks[0]


def _parse_snippets(body: bytes) -> list[tuple[float, str]]:
    """``(start_seconds, text)`` per non-empty ``<text>`` of a timedtext XML body."""
    try:
        root = ET.fromstring(body)
        snippets: list[tuple[float, str]] = []
        for element in root.iter("text"):
            # ElementTree decodes the XML layer; the text inside is still
            # HTML-escaped (`&amp;#39;` arrives as `&#39;`) and may hold tags.
            text = _HTML_TAG.sub("", html.unescape(element.text or ""))
            text = " ".join(text.split())
            if text:
                snippets.append((float(element.get("start", "0")), text))
    except (ET.ParseError, ValueError) as exc:
        raise ProviderError(f"youtube: unparseable transcript XML: {exc}") from exc
    return snippets


def _paragraphs(snippets: list[tuple[float, str]]) -> list[str]:
    """Join snippets into ``[m:ss] text`` paragraphs of at most a minute each."""
    paragraphs: list[str] = []
    start: float | None = None
    texts: list[str] = []
    for snippet_start, text in snippets:
        if start is None or snippet_start >= start + _PARAGRAPH_SECONDS:
            if start is not None:
                paragraphs.append(f"[{_clock(start)}] {' '.join(texts)}")
            start, texts = snippet_start, []
        texts.append(text)
    if start is not None:
        paragraphs.append(f"[{_clock(start)}] {' '.join(texts)}")
    return paragraphs


async def fetch_transcript(client: httpx.AsyncClient, video_id: str, retries: int) -> str:
    """Fetch the transcript of ``video_id`` and render it as Markdown.

    Raises ``ProviderError`` when the video is not playable (worded as bot
    protection when YouTube asks to confirm we are not a bot), has no captions
    (worded as an empty response), or when either answer cannot be parsed or
    the transcript holds no text.
    """
    response = await request_with_retry(
        client,
        "POST",
        PLAYER_ENDPOINT,
        retries=retries,
        provider=_PROVIDER,
        json={"context": {"client": _PLAYER_CLIENT}, "videoId": video_id},
    )
    try:
        player = response.json()
    except ValueError as exc:
        raise ProviderError(f"youtube: invalid player response: {exc}") from exc
    if not isinstance(player, dict):
        raise ProviderError("youtube: invalid player response: not a JSON object")

    playability = player.get("playabilityStatus") or {}
    status = playability.get("status")
    if status != "OK":
        reason = playability.get("reason") or ""
        if "not a bot" in reason.lower():
            # src.failure_reason.classify keys on the "bot protection" wording.
            raise ProviderError(f"youtube: blocked by bot protection ({status}: {reason})")
        raise ProviderError(f"youtube: video not playable ({status}: {reason})")

    renderer = (player.get("captions") or {}).get("playerCaptionsTracklistRenderer") or {}
    tracks = renderer.get("captionTracks") or []
    if not tracks:
        # "empty response" is the wording classify() maps to `empty`.
        raise ProviderError("youtube: empty response (the video has no captions)")
    track = _pick_track(player, tracks)

    # The srv3 format is a richer XML with per-word timing; without the
    # parameter the endpoint answers with the plain <transcript> shape.
    timedtext = await request_with_retry(
        client,
        "GET",
        track["baseUrl"].replace("&fmt=srv3", ""),
        retries=retries,
        provider=_PROVIDER,
    )
    snippets = _parse_snippets(timedtext.content)
    if not snippets:
        raise ProviderError("youtube: empty response (the caption track has no text)")

    details = player.get("videoDetails") or {}
    caption_name = "".join(run.get("text", "") for run in (track.get("name") or {}).get("runs", []))
    duration = _clock(float(details.get("lengthSeconds") or 0))
    lines = [
        f"# {details.get('title', '')}",
        "",
        f"Channel: {details.get('author', '')} · Duration: {duration} · Captions: {caption_name}",
        "",
    ]
    description = (details.get("shortDescription") or "").strip()
    if description:
        lines += ["## Description", "", description, ""]
    lines += ["## Transcript", "", "\n\n".join(_paragraphs(snippets))]
    return "\n".join(lines)
