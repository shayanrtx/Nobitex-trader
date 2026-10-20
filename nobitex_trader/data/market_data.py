"""Live market state: order book, candles and public trades.

Data arrives from the websocket when available; a REST poller takes over
automatically whenever the socket is down or silent, so the strategy never runs
on stale data without the risk layer knowing (``OrderBook.received_ts``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from nobitex_trader.domain.models import Candle, OrderBook, PublicTrade, Side
from nobitex_trader.exchange.base import MarketDataSource
from nobitex_trader.exchange.rest import parse_orderbook, to_decimal
from nobitex_trader.exchange.websocket import CentrifugoClient

log = logging.getLogger(__name__)

TradeListener = Callable[[PublicTrade], None]
BookListener = Callable[[OrderBook], None]


class CandleSeries:
    """Fixed-size candle history keyed by open time."""

    def __init__(self, max_len: int) -> None:
        self.max_len = max_len
        self._candles: list[Candle] = []

    def replace_all(self, candles: list[Candle]) -> None:
        self._candles = sorted(candles, key=lambda c: c.ts)[-self.max_len :]

    def upsert(self, candle: Candle) -> None:
        if self._candles and candle.ts == self._candles[-1].ts:
            self._candles[-1] = candle
        elif not self._candles or candle.ts > self._candles[-1].ts:
            self._candles.append(candle)
            if len(self._candles) > self.max_len:
                del self._candles[0]
        # Older-than-last updates are ignored (out-of-order delivery).

    def snapshot(self) -> list[Candle]:
        return list(self._candles)

    def __len__(self) -> int:
        return len(self._candles)


class MarketDataFeed:
    def __init__(
        self,
        symbol: str,
        source: MarketDataSource,
        ws: CentrifugoClient | None,
        candle_resolution: str,
        candle_history: int,
        poll_interval: float = 2.0,
        ws_stale_after: float = 15.0,
        candle_refresh_seconds: float = 60.0,
    ) -> None:
        self.symbol = symbol
        self.resolution = candle_resolution
        self._source = source
        self._ws = ws
        self._poll_interval = poll_interval
        self._ws_stale_after = ws_stale_after
        self._candle_refresh = candle_refresh_seconds
        self._history = candle_history
        self.candles = CandleSeries(candle_history)
        self.orderbook: OrderBook | None = None
        self.updated = asyncio.Event()
        self._last_ws_book = 0.0
        self._trade_listeners: list[TradeListener] = []
        self._book_listeners: list[BookListener] = []
        self._tasks: list[asyncio.Task[None]] = []

        if ws is not None:
            ws.subscribe(f"public:orderbook-{symbol}", self._on_ws_book)
            ws.subscribe(f"public:candle-{symbol}-{candle_resolution}", self._on_ws_candle)
            ws.subscribe(f"public:trades-{symbol}", self._on_ws_trade)

    def on_trade(self, listener: TradeListener) -> None:
        self._trade_listeners.append(listener)

    def on_book(self, listener: BookListener) -> None:
        self._book_listeners.append(listener)

    async def start(self) -> None:
        """Load initial snapshots (fail fast if the exchange is unreachable)."""
        await self.refresh_candles()
        self._set_book(await self._source.orderbook(self.symbol))
        self._tasks.append(asyncio.create_task(self._poll_loop(), name="md-poll"))
        self._tasks.append(asyncio.create_task(self._candle_loop(), name="md-candles"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    async def refresh_candles(self) -> None:
        candles = await self._source.candles(self.symbol, self.resolution, self._history)
        if candles:
            self.candles.replace_all(candles)

    # ----------------------------------------------------------------- internals

    def _set_book(self, book: OrderBook) -> None:
        self.orderbook = book
        for listener in self._book_listeners:
            try:
                listener(book)
            except Exception:
                log.exception("book listener failed")
        self.updated.set()

    def _ws_healthy(self) -> bool:
        return (
            self._ws is not None
            and self._ws.connected.is_set()
            and time.time() - self._last_ws_book < self._ws_stale_after
        )

    async def _poll_loop(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval)
            if self._ws_healthy():
                continue
            try:
                self._set_book(await self._source.orderbook(self.symbol))
            except Exception as exc:
                log.warning("orderbook poll failed: %r", exc)

    async def _candle_loop(self) -> None:
        # Periodic REST refresh keeps history correct even if candle pushes are missed.
        while True:
            await asyncio.sleep(self._candle_refresh)
            try:
                await self.refresh_candles()
            except Exception as exc:
                log.warning("candle refresh failed: %r", exc)

    def _on_ws_book(self, _channel: str, data: dict[str, Any]) -> None:
        self._last_ws_book = time.time()
        self._set_book(parse_orderbook(data, received_ts=self._last_ws_book))

    def _on_ws_candle(self, _channel: str, data: dict[str, Any]) -> None:
        try:
            self.candles.upsert(
                Candle(
                    ts=int(data["t"]),
                    open=to_decimal(data["o"]),
                    high=to_decimal(data["h"]),
                    low=to_decimal(data["l"]),
                    close=to_decimal(data["c"]),
                    volume=to_decimal(data.get("v")),
                )
            )
        except (KeyError, TypeError, ValueError):
            log.warning("bad candle payload: %s", data)

    def _on_ws_trade(self, _channel: str, data: dict[str, Any]) -> None:
        price, volume = to_decimal(data.get("price")), to_decimal(data.get("volume"))
        if price <= 0 or volume <= 0:
            return
        raw_ts = data.get("time") or time.time() * 1000
        trade = PublicTrade(
            price=price,
            volume=volume,
            side=Side.SELL if str(data.get("type")).lower() == "sell" else Side.BUY,
            ts=float(raw_ts) / 1000.0,
        )
        for listener in self._trade_listeners:
            try:
                listener(trade)
            except Exception:
                log.exception("trade listener failed")

    @property
    def mid(self) -> Decimal | None:
        return self.orderbook.mid if self.orderbook else None
