# Каталог тест-кейсов чтения (read_page / read_pages)

Набор реальных страниц, на которых стоит проверять движок чтения research-mcp:
плохо парсящиеся (бот-стены, кривые PDF, шум навигации, JS) и эталонно-чистые
(positive controls). Собрано из продакшн-отчётов об инструментах (исследование по
термопарам, 2026-06) и латентных кейсов из лога запросов.

Машиночитаемая версия — рядом в [`read_cases.yaml`](./read_cases.yaml) (источник
истины для будущей автоматизации). Этот `.md` — человекочитаемый рендер.

## Как использовать

Это **ручной / опциональный бенчмарк**, намеренно **не** включённый в CI: URL
живые, часть за Cloudflare/paywall, network недетерминирован — в CI это было бы
флакающим.

1. Прогнать `read_page` (или `read_pages`) по каждому URL.
2. Сверить результат с колонкой «Ожидаемо».
3. Параллельно смотреть строку в `data/research-mcp.log`: для провалов —
   `FAILED ok=false ... tried=[...]`; для успехов — `provider=... ok=true ...
   paid_calls=...`. Так видно, каким ярусом и за сколько прочиталось.

`url: null` в YAML / `—` в таблице = точный адрес в отчёте не зафиксирован,
вписать перед использованием кейса.

## Легенда статусов

| Статус | Значение |
|---|---|
| `unavoidable` | Бот-стена/paywall — автоматом не берётся, движок должен честно вернуть `ok=false` |
| `fixed:A` | Чинится фолбэком PDF при провале прямой пробы (PR #2) |
| `fixed:B` | Чинится ретраем без TLS-проверки на SSL-ошибке (PR #2) |
| `fixed:C` | Чинится `favor_precision` в trafilatura — срез навигации (PR #2) |
| `limitation` | Известное ограничение (JS-контент); может улучшиться через crawl4ai |
| `perf` | Читается, но медленно (глубокий фолбэк) — контроль латентности |
| `control` | Эталон: должен извлекаться чисто, иначе регресс |

## Кейсы

| id | URL | Тип | Категория | Статус | Проблема / Ожидаемо |
|---|---|---|---|---|---|
| wiley-thermocouple-insulation | — (onlinelibrary.wiley.com) | html | cloudflare-captcha | unavoidable | CAPTCHA вместо статьи → `ok=false` + FAILED-строка |
| researchgate-thermocouple-1000c | — (researchgate.net) | html | cloudflare-captcha | unavoidable | Security check → `ok=false` |
| control-com-forum-downward-spikes | — (control.com форум) | html | cloudflare-captcha | unavoidable | CAPTCHA → `ok=false` |
| keysight-thermocouple-error-sources | [docs.keysight.com](https://docs.keysight.com/kkbopen/what-are-the-source-of-error-in-thermocouple-measurements-985405150.html) | html | cloudflare-captcha | unavoidable | CAPTCHA (crawl4ai сам предупреждает) → `ok=false` |
| omega-z021-032-reference | — (mx.omega.com, PDF) | pdf | pdf-ssl | fixed:B | SSL CERTIFICATE_VERIFY_FAILED → ретрай без проверки + pypdf |
| aip-roberts-kollie-1977 | — (pubs.aip.org, PDF) | pdf | pdf-403 | fixed:A | 403 на PDF → фолбэк в jina/tavily/firecrawl (paywall может не пустить) |
| jms-se-insulation-resistance | — (jms-se.com, PDF) | pdf | pdf-empty | fixed:A | Пустой ответ → фолбэк в провайдеров |
| te-instrumentation-accuracy-classes | [te-instrumentation.com](https://www.te-instrumentation.com/understanding-thermocouple-accuracy-classes/) | html | nav-noise | fixed:C | ~15 КБ навигации вокруг ~2 КБ статьи → precision срезает меню |
| nanmac-thermocouple-article | — (nanmac.com) | html | nav-noise | fixed:C | Простыня меню/услуг перед статьёй → precision срезает |
| okazaki-thermocouple-faq | — (okazaki-mfg.com) | html | nav-noise | fixed:C | Дерево продуктов/FAQ перед контентом → precision срезает |
| jumo-15-thermocouple-problems | [jumo.group](https://www.jumo.group/nl/en/about-us/blog/thermocouple-errors) | html | js-rendered | limitation | JS-контент, извлеклось 2-3 предложения → эскалация на crawl4ai |
| firecrawl-blog-best-search-apis | [firecrawl.dev](https://www.firecrawl.dev/blog/best-web-search-apis) | html | slow-deep-fallback | perf | Дочитал tavily за ~127 с — контроль латентности |
| exa-blog-nextgen-search | [exa.ai](https://exa.ai/blog/how-to-build-nextgen-search) | html | slow-deep-fallback | perf | Дочитал tavily за ~127 с |
| perplexity-ai-first-search-api | [research.perplexity.ai](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api) | html | slow-deep-fallback | perf | jina за ~79 с |
| temperatures-ru | — (temperatures.ru) | html | control-clean | control | Эталон: чистый полный текст |
| owen-ru-tables | — (owen.ru) | html | control-clean | control | Эталон на таблицы (precision не должен их терять) |
| wika-blog | — (WIKA blog) | html | control-clean | control | Эталон: чистый текст |
| ht-heater-com | — (ht-heater.com) | html | control-clean | control | Эталон: практический контент |
| thermocouplechina-com | — (thermocouplechina.com) | html | control-clean | control | Эталон: техтекст |

## Доработать

- Вписать точные URL для `url: null` (wiley/researchgate/control/omega/aip/jms/
  nanmac/okazaki + контроли) — в отчётах были домены и описание, но не всегда
  полный адрес.
- При желании — небольшой opt-in скрипт-раннер, который гоняет `read_cases.yaml`
  через задеплоенный MCP и печатает provider/ok/elapsed/paid по каждому кейсу
  (в CI не включать).
