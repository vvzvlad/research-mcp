"""Provider plugins for the research facade.

A *type* is an implementation class (e.g. the ``searxng`` search provider); an
*instance* is a configured copy of a type with its credentials/URL resolved from
named environment variables (see ``src/pipeline_config.py``).

Importing this package imports every provider module so each ``@register(...)``
decorator runs and populates ``REGISTRY``.
"""

from src.providers import (  # noqa: F401  (imported for their @register side effects)
    crawl4ai,
    exa,
    firecrawl,
    jina,
    searxng,
    serper,
    tavily,
    trafilatura,
)
