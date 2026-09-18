"""XMLRiver SERP proxy provider (Yandex by default, Google optional).

API (verified against the official docs 2026-09-18,
https://xmlriver.com/api/api-connect/): GET
``https://xmlriver.com/search_yandex/xml?user=[user_id]&key=[key]&query=test``
for Yandex and ``https://xmlriver.com/search/xml?...`` for Google. Two things
make this provider unlike every other one here:

1. Authentication is a pair of QUERY parameters, not a header: ``user`` (the
   account id) and ``key``. Hence this instance needs BOTH ``token`` (the user
   id) and ``api_key`` (the key) from its config.
2. The answer is XML, "адаптирован к формату Yandex Search API"
   (https://xmlriver.com/api/api-answer/)::

       <yandexsearch version="1.0">
         <response date="...">
           <results><grouping>
             <group><doc>
               <url/><title/><passages><passage/></passages>
             </doc></group>
           </grouping></results>
         </response>
       </yandexsearch>

   The snippet is ``doc/passages/passage`` — possibly several per document —
   NOT a ``snippet`` tag; ``snippet`` exists only inside ``sitelinks/sitelink``.
   The vendor's own PHP sample (https://xmlriver.com/api/api-po/) still reads
   ``doc/snippet``, so that tag is kept as a fallback.

Errors arrive INSIDE the body, replacing the ``results`` block::

    <response date="..."><error code="15">...</error></response>

so they must be recognised in the XML or a failed query would look like an
empty SERP. https://xmlriver.com/api/api-errors/ lists the codes (2 empty
query, 42 wrong key, 45 IP not allowed, 102-107 bad parameter, 200 out of
funds, 500 network blip, ...); only code 115 is documented to carry an HTTP
status of its own (429). Code 15 is the one that is NOT a failure —
"Примечание. Если по поисковому запросу отсутствуют результаты, допустима
ошибка с кодом «15»" — i.e. a normal empty answer.

Parameters (https://xmlriver.com/apiydoc/apiy-about/ for Yandex,
https://xmlriver.com/apidoc/api-about/ for Google):

- ``page`` — "первая страница в Яндексе имеет номер 0, а в Google – 1", so the
  0/1 base depends on the engine.
- ``groupby`` (the TOP size) is deliberately NOT sent. Yandex documents the
  single value 10, and a TOP100 account setting applies ONLY while groupby is
  absent from the GET request ("при передаче GET-параметра groupby=100, но не
  установленном в кабинете ТОП100 ... ошибку 107"); for Google the parameter
  "с сентября 2025 года ... игнорируется и всегда равен 10". So ``num_results``
  cannot be expressed per request — the pipeline trims the merged list anyway,
  and omitting groupby lets the account's own setting stand.
- ``lang`` (Yandex) — "Код языка Яндекса: ru, uk ...": a bare two-letter code.
  No closed list is published and no error code covers an unknown value (102-106
  cover groupby/lr/loc/country/domain only), so an unsupported code is at worst
  ignored. Google's counterpart is ``lr``, a NUMERIC id from the vendor's
  langs.xlsx file, which we do not ship — so no language is sent for Google.

Engine choice: Yandex is the default on purpose. This provider exists for
Russian-language results, which is the only reason to pay for a Russian SERP
proxy; Google is already covered by serper. The class reads the engine from
``config.options["engine"]``, but nothing populates that today: ``Instance`` has
no options field and ``Pipeline.build`` fills ``options`` only for the jina
reader. So Google is reachable from tests only — switching production to it
needs those two plumbing changes first, not just a config line.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from src.providers._http import request_with_retry
from src.providers.base import ProviderConfig, ProviderError, SearchResult
from src.providers.registry import register

# The engine is selected by the endpoint path, not by a parameter.
XMLRIVER_ENDPOINTS = {
    "yandex": "https://xmlriver.com/search_yandex/xml",
    "google": "https://xmlriver.com/search/xml",
}

# Number of the FIRST result page per engine (documented explicitly: Yandex
# counts pages from zero, Google from one).
XMLRIVER_FIRST_PAGE = {"yandex": 0, "google": 1}

# "Nothing found for this query" — the one error code the docs call acceptable
# ("допустима ошибка с кодом «15»"), so it maps onto an empty result list.
XMLRIVER_NO_RESULTS_CODE = "15"


def _text(node: ET.Element | None) -> str:
    """Flatten an element's text content, inner markup included.

    With ``highlights=1`` XMLRiver wraps matched words of a title/passage in
    ``<hlword>`` tags. We never ask for that, but ``.text`` alone would stop at
    the first inner element, so read the whole subtree and squeeze the
    pretty-printer's newlines and indentation out of the result.
    """
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _yandex_lang(value: str) -> str | None:
    """Map a caller language tag onto a Yandex language code.

    ``language`` is whatever the LLM passed — ``ru-RU``, ``en_US``, ``RU``,
    ``" ru "`` — while the documented format is a bare two-letter code (``ru``,
    ``uk``, ...). So normalise separator and case, drop the region subtag, and
    return the result only if it really is two ASCII letters.

    Returns None otherwise: the full set of codes Yandex accepts is not
    published, so anything that does not even look like a language code is not
    worth sending.
    """
    code = value.strip().replace("_", "-").lower().split("-")[0]
    return code if len(code) == 2 and code.isascii() and code.isalpha() else None


@register("xmlriver_search")
class XmlRiverSearch:
    """Yandex/Google SERP via XMLRiver (requires ``token`` + ``api_key``)."""

    def __init__(self, config: ProviderConfig) -> None:
        # Two secrets, not one: `user` is the numeric account id and `key` the
        # API key, and both travel in the query string (see the module docstring).
        if not config.token:
            raise ValueError("xmlriver_search requires a token (the XMLRiver user id)")
        if not config.api_key:
            raise ValueError("xmlriver_search requires an api_key")
        engine = (config.options.get("engine") or "yandex").strip().lower()
        if engine not in XMLRIVER_ENDPOINTS:
            raise ValueError(
                f"xmlriver_search: unknown engine {engine!r} "
                f"(expected one of {sorted(XMLRIVER_ENDPOINTS)})"
            )
        self.name = config.name
        self.proxy = config.proxy
        self._config = config
        self._engine = engine
        self._endpoint = XMLRIVER_ENDPOINTS[engine]

    async def search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int,
        page: int,
        language: str | None,
    ) -> list[SearchResult]:
        # `num_results` cannot be passed on: the TOP size is fixed at 10 per
        # request by the vendor and `groupby` must stay absent — see the module
        # docstring. Paging is the only way deeper, and it is supported, so
        # page > 1 is a normal request here (no depth refusal like brave's).
        first_page = XMLRIVER_FIRST_PAGE[self._engine]
        params: dict[str, Any] = {
            "user": self._config.token,
            "key": self._config.api_key,
            # httpx percent-encodes the value, which also covers the vendor's
            # "амперсанд (&) ... необходимо заменять на код %26" rule.
            "query": query,
            # max(0, ...) covers page <= 0: nothing upstream clamps `page`
            # (src/server.py clamps only num_results).
            "page": first_page + max(0, page - 1),
        }
        if language and self._engine == "yandex":
            # Google's language parameter is `lr`, a numeric id from a vendor
            # file, so only Yandex gets a language here.
            lang = _yandex_lang(language)
            if lang:
                params["lang"] = lang
        response = await request_with_retry(
            client,
            "GET",
            self._endpoint,
            params=params,
            retries=self._config.retries,
            provider=self.name,
        )
        try:
            # Parse the BYTES, not response.text: that way the XML declaration's
            # own encoding decides, instead of a charset guessed from headers.
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ProviderError(f"{self.name}: invalid XML response") from exc
        node = root.find("response")
        if node is None:
            # Not an empty SERP — the envelope itself is not what the docs
            # describe, so treat it as a failure rather than "nothing found".
            raise ProviderError(f"{self.name}: unexpected XML (no <response> element)")
        error = node.find("error")
        if error is not None:
            # An error is served in the body (HTTP 200 for everything except the
            # documented 429 of code 115), so without this branch a wrong key or
            # an empty balance would be reported as a successful empty search.
            code = (error.get("code") or "").strip()
            if code == XMLRIVER_NO_RESULTS_CODE:
                return []
            raise ProviderError(
                f"{self.name}: xmlriver error {code or '?'}: "
                f"{_text(error) or 'no message'}"
            )
        out: list[SearchResult] = []
        for doc in node.iterfind("results/grouping/group/doc"):
            url = _text(doc.find("url"))
            if not url:
                continue
            # The snippet is one or more <passage> elements under <passages>.
            passages = (_text(p) for p in doc.iterfind("passages/passage"))
            snippet = " ".join(text for text in passages if text)
            if not snippet:
                # Fallback for the shape the vendor's own PHP sample reads.
                snippet = _text(doc.find("snippet"))
            out.append(
                SearchResult(
                    title=_text(doc.find("title")),
                    url=url,
                    snippet=snippet,
                    source=self.name,
                )
            )
        return out
