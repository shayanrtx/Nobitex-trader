from __future__ import annotations

import time
from decimal import Decimal

import pytest

from nobitex_trader.config import Settings
from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import Candle, OrderBook
from nobitex_trader.persistence.store import Store


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    for key in list(__import__("os").environ):
        if key.startswith("NBX_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NBX_CONFIG", str(tmp_path / "missing.yaml"))
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def market() -> MarketSpec:
    return MarketSpec(
        symbol="USDTIRT",
        src="usdt",
        dst="rls",
        price_tick=Decimal("10"),
        amount_step=Decimal("0.01"),
        min_notional=Decimal("3000000"),
        maker_fee=Decimal("0.0025"),
        taker_fee=Decimal("0.0025"),
    )


@pytest.fixture
def settings(tmp_path) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def store() -> Store:
    s = Store(":memory:")
    yield s  # type: ignore[misc]
    s.close()


def make_book(bid: str, ask: str, size: str = "1000", ts: float | None = None) -> OrderBook:
    now = ts if ts is not None else time.time()
    return OrderBook(
        bids=((Decimal(bid), Decimal(size)),),
        asks=((Decimal(ask), Decimal(size)),),
        last_trade_price=Decimal(bid),
        exchange_ts=now,
        received_ts=now,
    )


def flat_candles(n: int, price: int = 1_000_000, wiggle: int = 3000) -> list[Candle]:
    out = []
    for i in range(n):
        p = Decimal(price + (wiggle if i % 2 else -wiggle))
        out.append(Candle(1_700_000_000 + i * 900, p, p + wiggle, p - wiggle, p, Decimal(10)))
    return out
