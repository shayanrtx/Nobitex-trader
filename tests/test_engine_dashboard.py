from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import httpx

from nobitex_trader.config import Settings
from nobitex_trader.dashboard.server import create_app
from nobitex_trader.data.market_data import MarketDataFeed
from nobitex_trader.domain.models import Candle, OrderBook
from nobitex_trader.engine import TradingEngine
from nobitex_trader.exchange.paper import PaperVenue
from nobitex_trader.notify import Notifier
from nobitex_trader.persistence.store import Store
from nobitex_trader.strategy.registry import create_strategy
from tests.conftest import flat_candles

D = Decimal


class FakeSource:
    def __init__(self) -> None:
        self.mid = D(1_000_000)

    async def orderbook(self, symbol: str) -> OrderBook:
        now = time.time()
        return OrderBook(((self.mid - 1000, D(500)),), ((self.mid + 1000, D(500)),),
                         self.mid, now, now)

    async def candles(self, symbol, resolution, countback, to_ts=None) -> list[Candle]:  # type: ignore[no-untyped-def]
        return flat_candles(countback)


class RecordingNotifier(Notifier):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


async def test_engine_quotes_and_dashboard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(
        data_dir=tmp_path,
        engine={"tick_seconds": 0.05, "min_requote_seconds": 0.05, "order_sync_seconds": 0.05,
                "equity_snapshot_seconds": 0.1},
        paper={"initial_base": "250", "initial_quote": "250000000"},
        risk={"trading_windows": []},
        grid={"filled_order_delay_seconds": 0},
    )
    market = settings.market.spec()
    store = Store(settings.db_path)
    source = FakeSource()
    feed = MarketDataFeed(market.symbol, source, None, "15", 200, poll_interval=0.05)
    venue = PaperVenue(market, {"usdt": D(250), "rls": D(250_000_000)}, store=store)
    feed.on_book(venue.on_book)
    notifier = RecordingNotifier()
    engine = TradingEngine(settings, market, feed, venue, create_strategy(settings), store,
                           notifier)
    await feed.start()
    task = asyncio.create_task(engine.run())
    await asyncio.sleep(0.6)
    assert engine.oms.open_orders, engine.last_info
    assert any("شروع" in m for m in notifier.messages)

    app = create_app(engine, "secret")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        assert (await client.get("/api/summary")).status_code == 401
        h = {"Authorization": "Bearer secret"}
        summary = (await client.get("/api/summary", headers=h)).json()
        assert summary["mode"] == "paper" and summary["equity"] > 0
        orders = (await client.get("/api/orders", headers=h)).json()
        assert len(orders) == len(engine.oms.open_orders)
        events = (await client.get("/api/events", headers=h)).json()
        assert any(e["kind"] == "order" for e in events)
        assert (await client.get("/", headers=h)).status_code == 200

        # a price drop fills resting buys
        source.mid = D(980_000)
        await asyncio.sleep(0.6)
        fills = (await client.get("/api/fills", headers=h)).json()
        assert fills and fills[0]["side"] == "buy"
        assert (await client.get("/api/stats", headers=h)).json()
        assert (await client.get("/api/equity?hours=1", headers=h)).json()

        # pause → all orders cancelled
        assert (await client.post("/api/control/pause", headers=h)).json()["ok"]
        await asyncio.sleep(0.3)
        assert engine.oms.open_orders == []
        await client.post("/api/control/resume", headers=h)
        await asyncio.sleep(0.4)
        assert engine.oms.open_orders

    engine.stop()
    await asyncio.wait_for(task, timeout=5)
    await feed.stop()
    assert engine.oms.open_orders == []  # cancel_on_shutdown
    store.close()
