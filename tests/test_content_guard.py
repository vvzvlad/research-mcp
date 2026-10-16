"""The block-page detector: captcha interstitials (short pages only) and paywall
teasers (any length) vs real content.

Unit-level and pure — no network. The pipeline integration (a challenge page
fails the reader; a teaser is kept only as a flagged thin fallback) lives in
tests/test_pipeline.py.
"""

from __future__ import annotations

from src import failure_reason
from src.content_guard import MAX_BLOCK_PAGE_CHARS, detect_block


def test_captcha_page_is_bot_protection():
    page = "Just a moment...\n\nChecking your browser before accessing the site."
    assert detect_block(page) == failure_reason.BOT_PROTECTION


def test_paywall_teaser_is_access_denied():
    page = "Subscribe to unlock this article and keep reading the full story."
    assert detect_block(page) == failure_reason.ACCESS_DENIED


def test_marker_match_is_case_insensitive():
    assert detect_block("JUST A MOMENT while we verify you") == failure_reason.BOT_PROTECTION


def test_ordinary_short_paragraph_is_not_a_block():
    page = "Thermocouples measure temperature via the Seebeck effect across a junction."
    assert detect_block(page) is None


def test_long_article_quoting_a_captcha_phrase_passes_the_length_gate():
    # A challenge page is a short interstitial; a real article that says "just a
    # moment" is far longer, so the gate keeps it from being judged a challenge.
    article = "Just a moment of history: the thermocouple dates back to 1821. "
    article += "Background and analysis follows. " * 80
    assert len(article) >= MAX_BLOCK_PAGE_CHARS
    assert detect_block(article) is None


def test_long_paywall_teaser_is_access_denied_at_any_length():
    # The measured NYT shape: nav chrome and the opening paragraphs of the real
    # article, then the preview notice. Long, yet still a teaser.
    teaser = "Skip to content. Technology. " + "A real opening paragraph. " * 80
    teaser += "You have a preview view of this article while we are checking your access."
    assert len(teaser) >= MAX_BLOCK_PAGE_CHARS
    assert detect_block(teaser) == failure_reason.ACCESS_DENIED


def test_the_captcha_cap_is_the_exact_boundary():
    # At the cap it is served (returns None); one char under it is still screened.
    marker = "just a moment"
    at_cap = marker + "x" * (MAX_BLOCK_PAGE_CHARS - len(marker))
    assert len(at_cap) == MAX_BLOCK_PAGE_CHARS
    assert detect_block(at_cap) is None
    assert detect_block(at_cap[:-1]) == failure_reason.BOT_PROTECTION
