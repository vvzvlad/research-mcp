"""Scholarly OA full text: url detection, the DOI/arXiv recovery path, the pipeline path.

All network I/O is mocked with respx; ``extract_pdf_text`` is monkeypatched so
the tests need no real PDF bytes.
"""

import httpx
import pytest
import respx

from src import failure_reason
from src.pipeline import Pipeline
from src.providers import _url_guard, scholarly
from src.providers.base import ProviderError
from tests.conftest import _clear_provider_env

DOI = "10.1234/example"
DOI_URL = f"https://doi.org/{DOI}"
UNPAYWALL = f"{scholarly._UNPAYWALL_API}{DOI}"
EMAIL = "me@example.com"
ARXIV_ID = "2103.00001"

# The publisher landing page the generic chain reads when scholarly falls through.
LANDING_HTML = (
    "<html><head><title>Landing</title></head><body><article><h1>A Paper</h1>"
    + "<p>The landing page carries a long abstract of the paper in plain text. </p>" * 12
    + "</article></body></html>"
)

# A minimal JATS full-text document: one abstract paragraph, two body paragraphs,
# the second with an inline tag that must be dropped.
JATS = (
    "<article><front><article-meta>"
    "<abstract><p>The abstract text.</p></abstract>"
    "</article-meta></front>"
    "<body><sec><p>First body paragraph.</p>"
    "<p>Second <italic>body</italic> paragraph.</p></sec></body>"
    "</article>"
)


def _reader(make_config, **options):
    return scholarly.ScholarlyRead(make_config("scholarly", options=options))


# -- url detection -------------------------------------------------------------


@pytest.mark.parametrize(
    "url,email,ok",
    [
        (DOI_URL, EMAIL, True),
        # No email: Unpaywall cannot be asked, so the DOI goes to the generic chain.
        (DOI_URL, None, False),
        # A doi.org page that is not a DOI.
        ("https://doi.org/the-identifier/resources/handbook", EMAIL, False),
        # arXiv needs no email, by url or by its DOI.
        (f"https://arxiv.org/abs/{ARXIV_ID}", None, True),
        (f"https://doi.org/10.48550/arXiv.{ARXIV_ID}", None, True),
        ("https://example.com/paper", EMAIL, False),
    ],
)
def test_accepts(url, email, ok, make_config):
    reader = _reader(make_config, unpaywall_email=email) if email else _reader(make_config)
    assert reader.accepts(url) is ok


# -- DOI -----------------------------------------------------------------------


@respx.mock
async def test_doi_repository_pdf_returns_text_and_sends_the_email(monkeypatch, make_config):
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: "Extracted paper body.")
    answer = {
        "title": "A Paper",
        "oa_locations": [
            {
                "host_type": "repository",
                "url": "https://repo.example/record/1",
                "url_for_pdf": "https://repo.example/record/1.pdf",
                "version": "acceptedVersion",
            }
        ],
    }
    api = respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    pdf = respx.get("https://repo.example/record/1.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 fake")
    )

    async with httpx.AsyncClient() as client:
        markdown = await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    # The header names the copy actually read, which may not be the published version.
    assert markdown.startswith(
        f"# A Paper\nDOI: {DOI}\nSource: https://repo.example/record/1 (acceptedVersion)\n\n"
    )
    assert "Extracted paper body." in markdown
    assert api.calls.last.request.url.params["email"] == EMAIL
    assert pdf.call_count == 1


@pytest.mark.parametrize(
    "pmc_url",
    [
        # The shape Unpaywall actually answers with: digits, no `PMC` prefix.
        "https://www.ncbi.nlm.nih.gov/pmc/articles/7745181",
        "https://europepmc.org/articles/PMC7745181",
        "http://europepmc.org/pmc/articles/PMC7745181",
    ],
)
@respx.mock
async def test_doi_pmc_location_reads_europepmc_full_text_xml(pmc_url, make_config):
    # url_for_pdf is null here: the PMC url wins and becomes the Europe PMC XML.
    answer = {
        "title": "PMC Paper",
        "oa_locations": [{"host_type": "repository", "url": pmc_url, "url_for_pdf": None}],
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    epmc = respx.get(f"{scholarly._EUROPEPMC_API}PMC7745181/fullTextXML").mock(
        return_value=httpx.Response(200, text=JATS)
    )

    async with httpx.AsyncClient() as client:
        markdown = await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert epmc.call_count == 1
    assert "The abstract text." in markdown
    assert "First body paragraph." in markdown
    # The inline <italic> tag is dropped, the words kept.
    assert "Second body paragraph." in markdown


@respx.mock
async def test_doi_without_oa_locations_is_classified_empty(make_config):
    respx.get(UNPAYWALL).mock(
        return_value=httpx.Response(200, json={"title": "No OA", "oa_locations": []})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


def test_jats_blocks_nested_in_a_paragraph_are_not_glued_into_it():
    # Publishers nest tables, figures and supplements inside a <p>: the table cells must not
    # join the paragraph, and a caption's own <p> is emitted exactly once.
    xml = (
        "<article><body><sec><p>Before the table "
        "<table-wrap><caption><p>Table caption</p></caption>"
        "<table><tr><td>cell-a</td><td>cell-b</td></tr></table></table-wrap>"
        " and <fig><caption><p>Figure caption</p></caption></fig> after.</p>"
        "<p><supplementary-material><label>Additional file 1:</label>"
        "<caption><p>Figure S1.</p></caption></supplementary-material></p>"
        "</sec></body></article>"
    ).encode()

    paragraphs = scholarly._jats_text(xml).split("\n\n")

    assert paragraphs == [
        "Before the table and after.",
        "Table caption",
        "Figure caption",
        "Figure S1.",
    ]


@respx.mock
async def test_doi_repository_copy_is_tried_before_the_publisher(monkeypatch, make_config):
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: "Repository copy.")
    answer = {
        "oa_locations": [
            {"host_type": "publisher", "url_for_pdf": "https://pub.example/p.pdf"},
            {"host_type": "repository", "url_for_pdf": "https://repo.example/r.pdf"},
        ]
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    publisher = respx.get("https://pub.example/p.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )
    repository = respx.get("https://repo.example/r.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )

    async with httpx.AsyncClient() as client:
        await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert repository.call_count == 1
    assert publisher.call_count == 0


@respx.mock
async def test_doi_best_oa_location_outside_oa_locations_is_used(monkeypatch, make_config):
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: "Best copy.")
    answer = {
        "oa_locations": [{"host_type": "publisher", "url": "https://pub.example/landing"}],
        "best_oa_location": {"host_type": "publisher", "url_for_pdf": "https://best.example/b.pdf"},
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    best = respx.get("https://best.example/b.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )

    async with httpx.AsyncClient() as client:
        markdown = await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert best.call_count == 1
    # No title, no location url, no version: the source is the url fetched.
    assert markdown == f"DOI: {DOI}\nSource: https://best.example/b.pdf\n\nBest copy."


@respx.mock
async def test_doi_special_characters_are_escaped_in_the_unpaywall_path(make_config):
    # An old SICI-style DOI may hold `<`, `>` and `#`; unescaped, `#` would cut
    # the path into a fragment.
    api = respx.get(host="api.unpaywall.org").mock(
        return_value=httpx.Response(200, json={"oa_locations": []})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError):
            await _reader(make_config, unpaywall_email=EMAIL).read(
                client, "https://doi.org/10.1000/a%3Cb%3E%23c"
            )

    assert api.calls.last.request.url.raw_path.split(b"?")[0] == b"/v2/10.1000/a%3Cb%3E%23c"


@respx.mock
async def test_doi_same_target_from_two_locations_is_fetched_once(make_config):
    # One PMC id listed on ncbi and on Europe PMC resolves to one XML: after it
    # yields nothing, the second location does not fetch it again.
    answer = {
        "oa_locations": [
            {"host_type": "repository", "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/7745181"},
            {"host_type": "repository", "url": "https://europepmc.org/articles/PMC7745181"},
        ]
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    epmc = respx.get(f"{scholarly._EUROPEPMC_API}PMC7745181/fullTextXML").mock(
        return_value=httpx.Response(200, text="<article><front/></article>")
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert epmc.call_count == 1
    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


@respx.mock
async def test_doi_unpaywall_rejecting_the_email_names_it(make_config):
    # Unpaywall answers 422 to a placeholder address such as one @example.com.
    respx.get(UNPAYWALL).mock(
        return_value=httpx.Response(422, json={"message": "Please use your own email address"})
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await _reader(make_config, unpaywall_email=EMAIL).read(client, DOI_URL)

    assert "UNPAYWALL_EMAIL" in str(caught.value)
    assert failure_reason.classify(caught.value) == failure_reason.OTHER


# -- arXiv ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "url", [f"https://arxiv.org/abs/{ARXIV_ID}", f"https://doi.org/10.48550/arXiv.{ARXIV_ID}"]
)
@respx.mock
async def test_arxiv_url_fetches_and_extracts_the_pdf(url, monkeypatch, make_config):
    # An arXiv DOI goes straight to arXiv (Unpaywall answers 404 for those), so
    # no email is needed and Unpaywall is never asked.
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: "arXiv full text.")
    pdf = respx.get(f"{scholarly._ARXIV_PDF}{ARXIV_ID}").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )

    async with httpx.AsyncClient() as client:
        markdown = await _reader(make_config).read(client, url)

    assert pdf.call_count == 1
    assert markdown.startswith(f"# arXiv:{ARXIV_ID}\n\n")
    assert "arXiv full text." in markdown


@respx.mock
async def test_arxiv_404_is_the_papers_own_status(make_config):
    # Here the fetched url IS the paper, so its 404 is a fact about the target.
    respx.get(f"{scholarly._ARXIV_PDF}{ARXIV_ID}").mock(return_value=httpx.Response(404))

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await _reader(make_config).read(client, f"https://arxiv.org/abs/{ARXIV_ID}")

    assert failure_reason.classify(caught.value) == failure_reason.NOT_FOUND


@respx.mock
async def test_arxiv_pdf_without_text_layer_is_classified_empty(monkeypatch, make_config):
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: scholarly.NO_TEXT_LAYER_NOTICE)
    respx.get(f"{scholarly._ARXIV_PDF}{ARXIV_ID}").mock(
        return_value=httpx.Response(200, content=b"%PDF scan")
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as caught:
            await _reader(make_config).read(client, f"https://arxiv.org/abs/{ARXIV_ID}")

    assert failure_reason.classify(caught.value) == failure_reason.EMPTY


# -- Pipeline.read -------------------------------------------------------------


def _public_dns(monkeypatch) -> None:
    """Resolve every host to a public address, so the SSRF guard needs no DNS."""

    async def _fake_resolve(host: str) -> list[str]:
        return ["142.250.74.46"]

    monkeypatch.setattr(_url_guard, "_resolve_host", _fake_resolve)


@respx.mock
async def test_read_doi_url_is_served_by_scholarly_before_the_probe(monkeypatch, settings):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("UNPAYWALL_EMAIL", EMAIL)
    monkeypatch.setattr(
        scholarly, "extract_pdf_text", lambda data: "Full paper text from the repository PDF. " * 5
    )
    answer = {
        "title": "A Paper",
        "oa_locations": [
            {
                "host_type": "repository",
                "url": "https://repo.example/1",
                "url_for_pdf": "https://repo.example/1.pdf",
            }
        ],
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    respx.get("https://repo.example/1.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )
    probe = respx.get(DOI_URL).mock(return_value=httpx.Response(200, text="<html></html>"))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(DOI_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "scholarly"
    assert outcome.tried == ["scholarly"]
    assert outcome.failures == []
    assert "Full paper text" in outcome.markdown
    # Served before the probe, so the probe and the generic chain never ran.
    assert probe.call_count == 0


@respx.mock
async def test_read_doi_candidates_are_fetched_with_the_guarded_client(monkeypatch, settings):
    # A candidate url comes from Unpaywall's JSON, so a hop into loopback must be
    # stopped by the SSRF guard: that candidate is skipped and the next one wins.
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("UNPAYWALL_EMAIL", EMAIL)
    monkeypatch.setattr(scholarly, "extract_pdf_text", lambda data: "Second repository copy.")
    answer = {
        "oa_locations": [
            {"host_type": "repository", "url_for_pdf": "https://repo1.example/a.pdf"},
            {"host_type": "repository", "url_for_pdf": "https://repo2.example/b.pdf"},
        ]
    }
    respx.get(UNPAYWALL).mock(return_value=httpx.Response(200, json=answer))
    respx.get("https://repo1.example/a.pdf").mock(
        return_value=httpx.Response(302, headers={"Location": "http://127.0.0.1/x"})
    )
    loopback = respx.get("http://127.0.0.1/x").mock(
        return_value=httpx.Response(200, content=b"%PDF internal")
    )
    second = respx.get("https://repo2.example/b.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF fake")
    )

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(DOI_URL)
    finally:
        await pipe.aclose()

    assert loopback.call_count == 0  # the hop never left the process
    assert second.call_count == 1
    assert outcome.provider == "scholarly"
    assert "Second repository copy." in outcome.markdown


@pytest.mark.parametrize(
    "unpaywall",
    [
        # Only a publisher landing page: no target, and the landing url is not fetched.
        httpx.Response(
            200,
            json={"oa_locations": [{"host_type": "publisher", "url": "https://pub.example/p"}]},
        ),
        # Unpaywall has no record: its own 404 says nothing about the paper's page.
        httpx.Response(404),
    ],
)
@respx.mock
async def test_read_doi_without_oa_copy_falls_through_to_the_generic_chain(
    unpaywall, monkeypatch, settings
):
    _clear_provider_env(monkeypatch)
    _public_dns(monkeypatch)
    monkeypatch.setenv("UNPAYWALL_EMAIL", EMAIL)
    respx.get(UNPAYWALL).mock(return_value=unpaywall)
    probe = respx.get(DOI_URL).mock(return_value=httpx.Response(200, text=LANDING_HTML))

    pipe = Pipeline.build(settings)
    try:
        outcome = await pipe.read(DOI_URL)
    finally:
        await pipe.aclose()

    assert outcome.provider == "trafilatura"
    assert outcome.tried == ["scholarly", "trafilatura"]
    assert outcome.failures == [("scholarly", failure_reason.EMPTY)]
    assert probe.call_count == 1
    assert "long abstract of the paper" in outcome.markdown
