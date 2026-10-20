"""Exchange-facing interfaces.

The engine only depends on these protocols, so the live REST client, the paper
simulator and the backtester are interchangeable.
"""

from __future__ import annotations

from typing import Protocol

from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import Balance, Candle, Order, OrderBook, OrderRequest


class MarketDataSource(Protocol):
    async def orderbook(self, symbol: str) -> OrderBook: ...

    async def candles(
        self, symbol: str, resolution: str, countback: int, to_ts: int | None = None
    ) -> list[Candle]: ...


class TradingVenue(Protocol):
    async def place_order(self, market: MarketSpec, request: OrderRequest) -> Order: ...

    async def cancel_order(self, market: MarketSpec, order: Order) -> Order | None:
        """Cancel and return the final order state (``None`` if unknown)."""
        ...

    async def get_order(
        self, market: MarketSpec, order_id: int | None = None, client_order_id: str | None = None
    ) -> Order: ...

    async def open_orders(self, market: MarketSpec) -> list[Order]: ...

    async def balances(self, currencies: list[str]) -> dict[str, Balance]: ...
