"""Strategy interface.

A strategy is a *pure function of market state*: given a context it returns the set
of resting orders it wants on the book. It never talks to the exchange. The order
manager diffs desired vs. actual orders, and the risk manager can veto or trim
quotes. This keeps strategies trivially testable and identical across live, paper
and backtest modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import Candle, Fill, OrderBook, Side


@dataclass(frozen=True, slots=True)
class Quote:
    side: Side
    price: Decimal
    amount: Decimal
    tag: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    base: Decimal  # bot-owned base currency (e.g. USDT), including amounts in open orders
    quote: Decimal  # bot-owned quote currency (e.g. rial), including amounts in open orders
    avg_cost: Decimal | None  # average acquisition price of ``base`` incl. fees

    def base_ratio(self, price: Decimal) -> Decimal:
        equity = self.quote + self.base * price
        if equity <= 0:
            return Decimal(0)
        return self.base * price / equity

    def equity(self, price: Decimal) -> Decimal:
        return self.quote + self.base * price


@dataclass(frozen=True, slots=True)
class StrategyContext:
    market: MarketSpec
    book: OrderBook
    candles: Sequence[Candle]
    inventory: InventorySnapshot
    now: float
    # Recent (timestamp, mid price) samples, oldest first — for short-term signals.
    price_history: Sequence[tuple[float, Decimal]] = ()


@dataclass(slots=True)
class StrategyDecision:
    quotes: list[Quote]
    # Free-form diagnostics for logs/status (fair price, spacing, targets ...).
    info: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def warmup_candles(self) -> int:
        """Candles needed before the strategy can quote."""

    @abstractmethod
    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        """Return the desired set of resting orders."""

    def on_fill(self, fill: Fill) -> None:  # noqa: B027 - optional hook
        """Optional hook for stateful strategies."""
