"""Provider type registry.

Each provider module decorates its class with ``@register("type")`` so the
instance loader can look the class up by its type name. ``REGISTRY`` maps the
type string to the implementation class.
"""

from __future__ import annotations

from typing import Callable, TypeVar

REGISTRY: dict[str, type] = {}

T = TypeVar("T", bound=type)


def register(type_name: str) -> Callable[[T], T]:
    """Class decorator: register the provider class under ``type_name``."""

    def _decorator(cls: T) -> T:
        if type_name in REGISTRY:
            raise ValueError(f"Provider type already registered: {type_name!r}")
        REGISTRY[type_name] = cls
        return cls

    return _decorator
