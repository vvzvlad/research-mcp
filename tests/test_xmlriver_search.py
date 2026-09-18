"""XMLRiver search provider: XML parsing, in-body errors, request shape.

The payloads mirror the Yandex-Search-API-compatible envelope documented on
2026-09-18 at https://xmlriver.com/api/api-answer/ (``yandexsearch`` →
``response`` → ``results/grouping/group/doc`` with ``url`` / ``title`` /
``passages/passage``, or an ``error`` element instead of ``results``). Network
I/O is mocked with respx, like the rest of the suite.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.providers.base import ProviderError
from src.providers.xmlriver_search import (
    XMLRIVER_ENDPOINTS,
    XMLRIVER_NO_RESULTS_CODE,
    XmlRiverSearch,
)

YANDEX_ENDPOINT = XMLRIVER_ENDPOINTS["yandex"]
GOOGLE_ENDPOINT = XMLRIVER_ENDPOINTS["google"]

# Two hits: the first also carries a <sitelink> whose own <snippet> must NOT be
# mistaken for the document's snippet, the second has two <passage> elements.
XMLRIVER_PAYLOAD = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response date="20260918T101500">
    <found priority="all">1200</found>
    <results>
      <grouping>
        <page first="1" last="10">0</page>
        <group>
          <doccount>1</doccount>
          <doc>
            <url>https://xmlriver.test/1</url>
            <title>Первый результат</title>
            <pubDate>18 сен. 2026 г. -</pubDate>
            <passages>
              <passage>сниппет один</passage>
            </passages>
            <sitelinks>
              <sitelink>
                <url>https://xmlriver.test/1/about</url>
                <title>О нас</title>
                <snippet>ссылка, а не сниппет документа</snippet>
              </sitelink>
            </sitelinks>
          </doc>
        </group>
        <group>
          <doccount>1</doccount>
          <doc>
            <url>https://xmlriver.test/2</url>
            <title>Второй результат</title>
            <passages>
              <passage>сниппет</passage>
              <passage>два</passage>
            </passages>
          </doc>
        </group>
      </grouping>
    </results>
  </response>
</yandexsearch>
"""

EMPTY_PAYLOAD = """<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response date="20260918T101500">
    <found priority="all">0</found>
    <results><grouping><page first="0" last="0">0</page></grouping></results>
  </response>
</yandexsearch>
"""


def _error_payload(code: str, message: str) -> bytes:
    """An XMLRiver failure: the ``error`` element replaces ``results``."""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<yandexsearch version="1.0">'
        '<response date="20260918T101500">'
        f'<error code="{code}">{message}</error>'
        "</response></yandexsearch>"
    ).encode()


def _xml_response(payload: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, content=payload.encode())


def _config(make_config, **kwargs):
    """Config with both secrets the provider needs: user id + key."""
    return make_config("xmlriver", token="42", api_key="secret", **kwargs)


# -- response parsing ------------------------------------------------------


@respx.mock
async def test_parses_documents_with_passages_as_snippet(make_config):
    # The snippet is doc/passages/passage (joined when there are several), NOT
    # the <snippet> that lives inside a sitelink.
    respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 10, 1, None)
    assert [(r.title, r.url, r.snippet, r.source) for r in results] == [
        ("Первый результат", "https://xmlriver.test/1", "сниппет один", "xmlriver"),
        ("Второй результат", "https://xmlriver.test/2", "сниппет два", "xmlriver"),
    ]


@respx.mock
async def test_falls_back_to_the_snippet_tag_of_the_vendor_sample(make_config):
    # The vendor's own PHP example reads doc/snippet; honour it when no
    # <passages> block is present.
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=_xml_response(
            '<?xml version="1.0" encoding="utf-8"?><yandexsearch version="1.0">'
            '<response date="x"><results><grouping><group><doc>'
            "<url>https://xmlriver.test/9</url><title>Заголовок</title>"
            "<snippet>старый формат</snippet>"
            "</doc></group></grouping></results></response></yandexsearch>"
        )
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 10, 1, None)
    assert [r.snippet for r in results] == ["старый формат"]


@respx.mock
async def test_inline_markup_inside_title_and_passage_is_flattened(make_config):
    # With highlights enabled on the account, matched words come wrapped in
    # <hlword>; reading only `.text` would truncate at the first tag.
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=_xml_response(
            '<?xml version="1.0" encoding="utf-8"?><yandexsearch version="1.0">'
            '<response date="x"><results><grouping><group><doc>'
            "<url>https://xmlriver.test/3</url>"
            "<title>Купить <hlword>слона</hlword> недорого</title>"
            "<passages><passage>Большой <hlword>слон</hlword> в наличии</passage></passages>"
            "</doc></group></grouping></results></response></yandexsearch>"
        )
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 10, 1, None)
    assert [(r.title, r.snippet) for r in results] == [
        ("Купить слона недорого", "Большой слон в наличии")
    ]


@respx.mock
async def test_grouping_without_documents_is_a_normal_empty_answer(make_config):
    respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(EMPTY_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 10, 1, None) == []


@respx.mock
async def test_documents_without_url_are_skipped(make_config):
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=_xml_response(
            '<?xml version="1.0" encoding="utf-8"?><yandexsearch version="1.0">'
            '<response date="x"><results><grouping>'
            "<group><doc><title>Без адреса</title></doc></group>"
            "<group><doc><url>https://xmlriver.test/ok</url><title>Есть</title></doc></group>"
            "</grouping></results></response></yandexsearch>"
        )
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 10, 1, None)
    assert [r.url for r in results] == ["https://xmlriver.test/ok"]


# -- request shape ---------------------------------------------------------


@respx.mock
async def test_credentials_travel_as_query_parameters(make_config):
    # Unlike every other provider here, XMLRiver authenticates with a pair of
    # GET parameters: `user` (account id) and `key`.
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "купить слона", 10, 1, None)
    request = route.calls.last.request
    assert request.method == "GET"
    assert str(request.url).startswith(YANDEX_ENDPOINT + "?")
    assert request.url.params["user"] == "42"
    assert request.url.params["key"] == "secret"
    assert request.url.params["query"] == "купить слона"
    # `groupby` is never sent: it would override the account's TOP setting and
    # only 10 is valid for Yandex anyway.
    assert "groupby" not in request.url.params
    assert "lang" not in request.url.params  # no language → not sent


@respx.mock
async def test_ampersand_in_the_query_is_percent_encoded(make_config):
    # The docs require "&" inside `query` to travel as %26; httpx's encoding of
    # the parameter value already does that.
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "rock & roll", 10, 1, None)
    request = route.calls.last.request
    assert "%26" in str(request.url)
    assert request.url.params["query"] == "rock & roll"


@respx.mock
@pytest.mark.parametrize(("page", "expected"), [(1, "0"), (3, "2"), (0, "0"), (-4, "0")])
async def test_yandex_pages_are_numbered_from_zero(make_config, page, expected):
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, page, None)
    assert route.calls.last.request.url.params["page"] == expected


@respx.mock
@pytest.mark.parametrize(("page", "expected"), [(1, "1"), (3, "3"), (0, "1")])
async def test_google_pages_are_numbered_from_one(make_config, page, expected):
    # Same provider, other engine: the docs state Google's first page is 1.
    route = respx.get(GOOGLE_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config, options={"engine": "google"}))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, page, None)
    assert route.calls.last.request.url.params["page"] == expected


@respx.mock
async def test_yandex_is_the_default_engine(make_config):
    # Russian results are the reason this provider exists, so the Yandex
    # endpoint is used unless the instance asks for google.
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, 1, None)
    assert route.call_count == 1


@respx.mock
@pytest.mark.parametrize(
    ("language", "expected"), [("ru-RU", "ru"), ("EN", "en"), ("  uk  ", "uk"), ("be_BY", "be")]
)
async def test_language_is_reduced_to_a_yandex_language_code(make_config, language, expected):
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, 1, language)
    assert route.calls.last.request.url.params["lang"] == expected


@respx.mock
@pytest.mark.parametrize("language", ["klingon", "x", "12", "", "   ", None])
async def test_language_that_is_not_a_two_letter_code_is_omitted(make_config, language):
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, 1, language)
    assert "lang" not in route.calls.last.request.url.params


@respx.mock
async def test_google_gets_no_language_parameter(make_config):
    # Google's counterpart is `lr`, a numeric id from a vendor file, so a
    # two-letter code must not be sent there.
    route = respx.get(GOOGLE_ENDPOINT).mock(return_value=_xml_response(XMLRIVER_PAYLOAD))
    provider = XmlRiverSearch(_config(make_config, options={"engine": "google"}))
    async with httpx.AsyncClient() as client:
        await provider.search(client, "q", 10, 1, "ru")
    params = route.calls.last.request.url.params
    assert "lang" not in params
    assert "lr" not in params


# -- errors carried inside a 200 response ----------------------------------


@respx.mock
async def test_error_in_the_body_of_a_200_is_a_provider_error(make_config):
    # HTTP is 200, the failure is only visible in the XML — without this the
    # pipeline would record a successful, empty search.
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            content=_error_payload("42", "Указан неверный ключ, выданный при регистрации"),
        )
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 10, 1, None)
    message = str(excinfo.value)
    assert "42" in message
    assert "неверный ключ" in message


@respx.mock
async def test_no_results_error_code_is_an_empty_answer(make_config):
    # Code 15 is documented as the acceptable "nothing found" answer, so it must
    # not fail the provider.
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            content=_error_payload(
                XMLRIVER_NO_RESULTS_CODE, "Искомая комбинация слов нигде не встречается"
            ),
        )
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        assert await provider.search(client, "q", 10, 1, None) == []


# -- transport failures ----------------------------------------------------


@respx.mock
async def test_payment_required_is_a_provider_error(make_config):
    route = respx.get(YANDEX_ENDPOINT).mock(return_value=httpx.Response(402))
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 10, 1, None)
    assert "out of credits" in str(excinfo.value)
    assert route.call_count == 1  # 402 is not retried


@respx.mock
async def test_malformed_body_is_a_provider_error(make_config):
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"<yandexsearch><response>")
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 10, 1, None)
    assert "invalid XML" in str(excinfo.value)


@respx.mock
async def test_envelope_without_a_response_element_is_a_provider_error(make_config):
    respx.get(YANDEX_ENDPOINT).mock(
        return_value=httpx.Response(200, content=b"<yandexsearch version='1.0'/>")
    )
    provider = XmlRiverSearch(_config(make_config))
    async with httpx.AsyncClient() as client:
        with pytest.raises(ProviderError) as excinfo:
            await provider.search(client, "q", 10, 1, None)
    assert "unexpected XML" in str(excinfo.value)


@respx.mock
async def test_server_error_is_retried_per_config(make_config):
    # retries comes from the config (1 in tests), so one 5xx is survivable.
    route = respx.get(YANDEX_ENDPOINT).mock(
        side_effect=[httpx.Response(500), _xml_response(XMLRIVER_PAYLOAD)]
    )
    config = _config(make_config)
    assert config.retries == 1
    provider = XmlRiverSearch(config)
    async with httpx.AsyncClient() as client:
        results = await provider.search(client, "q", 10, 1, None)
    assert len(results) == 2
    assert route.call_count == 2


# -- construction ----------------------------------------------------------


def test_requires_both_the_user_id_and_the_key(make_config):
    with pytest.raises(ValueError):
        XmlRiverSearch(make_config("xmlriver", api_key="secret"))  # no user id
    with pytest.raises(ValueError):
        XmlRiverSearch(make_config("xmlriver", token="42"))  # no key


def test_unknown_engine_is_rejected(make_config):
    with pytest.raises(ValueError):
        XmlRiverSearch(_config(make_config, options={"engine": "bing"}))
