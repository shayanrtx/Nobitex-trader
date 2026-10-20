"""Strategy registry — add new strategies here to make them selectable by name."""

from __future__ import annotations

from collections.abc import Callable

from nobitex_trader.config import Settings
from nobitex_trader.strategy.adaptive_grid import AdaptiveGridStrategy
from nobitex_trader.strategy.base import Strategy

_REGISTRY: dict[str, Callable[[Settings], Strategy]] = {
    AdaptiveGridStrategy.name: lambda s: AdaptiveGridStrategy(s.grid),
}


def register(name: str, factory: Callable[[Settings], Strategy]) -> None:
    _REGISTRY[name] = factory


def create_strategy(settings: Settings) -> Strategy:
    try:
        factory = _REGISTRY[settings.strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {settings.strategy!r}; available: {sorted(_REGISTRY)}"
        ) from None
    return factory(settings)


def available() -> list[str]:
    return sorted(_REGISTRY)
