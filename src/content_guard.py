"""Detect a block page a generic reader returned as if it were the article.

A reader can hand back a bot-challenge ("Just a moment…") or a paywall teaser
("Subscribe to unlock this article") as ``content``; served as the page body it
misleads the model into taking the wall for the article. ``detect_block`` names
which of the two it is, so the pipeline can keep looking for the full page: a
challenge page is dropped, a teaser (which carries real partial text) is kept
only as a last resort.

The two kinds are matched differently. A challenge page is a short interstitial,
so the captcha markers are tested only on pages under ``MAX_BLOCK_PAGE_CHARS``.
A paywall teaser can be long — nav chrome plus the first paragraphs of the real
article before the subscription notice — so the paywall markers, which are UI
strings rather than prose, are matched at any length.

Pure and I/O-free (like ``src/formatting.py`` / ``src/failure_reason.py``): the
only input is the markdown, the only output is a ``src.failure_reason`` category
or ``None``.

The markers are deliberately specific full phrases (in the spirit of
``_CREDIT_MARKERS`` in ``src/providers/_http.py``): a page that merely *discusses*
paywalls or captchas must not be caught. For the paywall markers that wording is
the only guard, since length is not; a page wrongly taken for a teaser still
reaches the model as the last-resort answer when no other reader does better.
Matched case-insensitively.
"""

from __future__ import annotations

from src import failure_reason

# Gate for the captcha markers only: a challenge page is a short interstitial,
# while a real article that happens to say "just a moment" is far longer, so a
# page this long or longer is never judged a challenge. The paywall markers
# ignore it (see the module docstring).
MAX_BLOCK_PAGE_CHARS = 1500

# CAPTCHA / bot-challenge interstitials. Cloudflare's "One more step" page always
# also carries "please complete the security check", so that common phrase is
# not a marker of its own.
_CAPTCHA_MARKERS = (
    "just a moment",
    "please complete the security check",
    "verifying you are human",
    "are you a robot",
)

# Paywall / subscription walls.
_PAYWALL_MARKERS = (
    "subscribe to unlock this article",
    "preview view of this article",
    "create an account to read the full story",
    "to read the full article subscribe",
    "this content is available to subscribers",
)


def detect_block(markdown: str) -> str | None:
    """Return the block category of ``markdown``, or ``None`` if it is not one.

    ``failure_reason.BOT_PROTECTION`` for a captcha/bot-challenge page shorter
    than ``MAX_BLOCK_PAGE_CHARS``, ``failure_reason.ACCESS_DENIED`` for a page of
    any length that carries a paywall marker, ``None`` for everything else. A
    short page with both kinds of marker is a challenge page.
    """
    lowered = markdown.lower()
    if len(markdown) < MAX_BLOCK_PAGE_CHARS and any(
        marker in lowered for marker in _CAPTCHA_MARKERS
    ):
        return failure_reason.BOT_PROTECTION
    if any(marker in lowered for marker in _PAYWALL_MARKERS):
        return failure_reason.ACCESS_DENIED
    return None
