from __future__ import annotations

import time
from decimal import Decimal

from nobitex_trader.config import GridSettings, RiskSettings
from nobitex_trader.domain.models import Fill, Side
from nobitex_trader.portfolio import Portfolio
from nobitex_trader.risk import RiskManager
from nobitex_trader.strategy.adaptive_grid import (
    AdaptiveGridStrategy,
    book_imbalance,
    short_term_return,
)
from nobitex_trader.strategy.base import InventorySnapshot, Quote, StrategyContext
from nobitex_trader.strategy.indicators import atr, ema
from tests.conftest import flat_candles, make_book

D = Decimal


def ctx(market, inv, book=None, candles=None, history=()):  # type: ignore[no-untyped-def]
    return StrategyContext(
        market=market,
        book=book or make_book("999000", "1001000"),
        candles=candles if candles is not None else flat_candles(150),
        inventory=inv,
        now=time.time(),
        price_history=history,
    )


def test_indicators() -> None:
    assert ema([D(1), D(2), D(3)], 3) == [D(2)]
    assert ema([D(1)], 3) == []
    candles = flat_candles(30, wiggle=1000)
    assert atr(candles, 14) is not None and atr(candles, 14) > 0


def test_warmup_returns_no_quotes(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings())
    d = s.decide(ctx(market, InventorySnapshot(D(0), D(10**9), None), candles=flat_candles(10)))
    assert d.quotes == [] and d.info["state"] == "warming_up"


def test_balanced_grid_is_symmetric_and_maker(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings(sell_above_cost=False, imbalance_weight=D(0)))
    inv = InventorySnapshot(base=D(250), quote=D(250_000_000), avg_cost=D(1_000_000))
    d = s.decide(ctx(market, inv))
    buys = [q for q in d.quotes if q.side is Side.BUY]
    sells = [q for q in d.quotes if q.side is Side.SELL]
    assert len(buys) == 4 and len(sells) == 4
    assert max(q.price for q in buys) < D(1001000)  # never crosses best ask
    assert min(q.price for q in sells) > D(999000)  # never crosses best bid
    for q in d.quotes:
        assert q.price % market.price_tick == 0
        assert q.notional >= market.min_notional
    # spacing between first buy and first sell >= minimum spacing (0.7%)
    gap = (min(q.price for q in sells) - max(q.price for q in buys)) / D(1_000_000)
    assert gap >= D("0.0069")


def test_inventory_skew(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings(sell_above_cost=False, imbalance_weight=D(0)))
    heavy = s.decide(ctx(market, InventorySnapshot(D(400), D(100_000_000), D(10**6))))
    light = s.decide(ctx(market, InventorySnapshot(D(100), D(400_000_000), D(10**6))))
    assert heavy.info["reservation"] < light.info["reservation"]
    heavy_sell = sum(q.amount for q in heavy.quotes if q.side is Side.SELL)
    light_sell = sum(q.amount for q in light.quotes if q.side is Side.SELL)
    assert heavy_sell > light_sell


def test_inventory_bounds_disable_a_side(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings())
    all_usdt = s.decide(ctx(market, InventorySnapshot(D(500), D(0), D(10**6))))
    assert all(q.side is Side.SELL for q in all_usdt.quotes)
    all_rial = s.decide(ctx(market, InventorySnapshot(D(0), D(500_000_000), None)))
    assert all(q.side is Side.BUY for q in all_rial.quotes)


def test_sell_floor_protects_cost(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings(sell_above_cost=True))
    inv = InventorySnapshot(D(300), D(200_000_000), avg_cost=D(1_050_000))
    d = s.decide(ctx(market, inv))
    floor = D(1_050_000) * (1 + D("0.0025") + D("0.002"))
    assert all(q.price >= floor for q in d.quotes if q.side is Side.SELL)


def test_momentum_guard_pauses_buys_in_a_crash(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings())
    now = time.time()
    history = [(now - 170 + i * 10, D(1_030_000) - D(1500) * i) for i in range(18)]
    inv = InventorySnapshot(D(250), D(250_000_000), D(10**6))
    d = s.decide(StrategyContext(market, make_book("999000", "1001000"), flat_candles(150),
                                 inv, now, tuple(history)))
    assert d.info["guard"] and not any(q.side is Side.BUY for q in d.quotes)


def test_signal_helpers() -> None:
    bids = [(D(100), D(30))]
    asks = [(D(101), D(10))]
    assert book_imbalance(bids, asks, 5) > D("0.4")
    now = 1000.0
    assert short_term_return([(now - 100, D(100)), (now, D(98))], now, 120) == D("-0.02")
    assert short_term_return([(now - 10, D(100)), (now, D(98))], now, 120) is None


def test_portfolio_average_cost_and_pnl(store) -> None:  # type: ignore[no-untyped-def]
    p = Portfolio(store, "usdt", "rls")
    p.initialize(D(0), D(100_000_000), D(1_000_000))
    p.apply_fill(Fill("a", 1, Side.BUY, D(1_000_000), D(10), D("0.025"), "usdt", 0))
    assert p.state.base == D("9.975")
    assert p.state.quote == D(90_000_000)
    assert p.state.avg_cost == D(10_000_000) / D("9.975")
    pnl = p.apply_fill(Fill("b", 2, Side.SELL, D(1_020_000), D("9.975"), D(25436), "rls", 1))
    expected = D(1_020_000) * D("9.975") - D(25436) - D(10_000_000)
    assert pnl == expected
    assert p.state.base == 0 and p.state.avg_cost is None
    # persisted & reloadable
    p2 = Portfolio(store, "usdt", "rls")
    assert p2.load() and p2.state.realized_pnl == expected


def test_risk_circuit_breaker_and_staleness(store) -> None:  # type: ignore[no-untyped-def]
    now = [10_000.0]
    rs = RiskSettings(circuit_breaker_move_pct=D(2), trading_windows=[])
    r = RiskManager(rs, store, clock=lambda: now[0])
    assert r.evaluate(make_book("1000000", "1001000", ts=now[0]), D(10**9)).allow_trading
    now[0] += 30
    stale = r.evaluate(make_book("1000000", "1001000", ts=now[0] - 60), D(10**9))
    assert not stale.allow_trading and "stale" in stale.reason
    crash = r.evaluate(make_book("970000", "971000", ts=now[0]), D(10**9))
    assert not crash.allow_trading and crash.new_event
    now[0] += 60
    assert "cooldown" in r.evaluate(make_book("970000", "971000", ts=now[0]), D(10**9)).reason


def test_risk_daily_loss_and_drawdown(store) -> None:  # type: ignore[no-untyped-def]
    now = [1_700_000_000.0]
    rs = RiskSettings(max_daily_loss_pct=D(3), max_drawdown_pct=D(5), trading_windows=[])
    r = RiskManager(rs, store, clock=lambda: now[0])
    book = lambda: make_book("1000000", "1001000", ts=now[0])  # noqa: E731
    assert r.evaluate(book(), D(1000)).allow_trading
    d = r.evaluate(book(), D(965))
    assert not d.allow_trading and "daily" in d.reason
    d = r.evaluate(book(), D(940))
    assert d.halted
    r.resume()
    assert r.halted_reason is None
    # resume clears the halt, but today's loss is still beyond the daily limit
    d = r.evaluate(book(), D(940))
    assert not d.halted and "daily" in d.reason
    now[0] += 86400  # next day: fresh day-start equity → trading allowed again
    assert r.evaluate(book(), D(940)).allow_trading


def test_risk_filters_quotes(store) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(RiskSettings(max_open_orders=2, max_price_deviation_pct=D(5)), store)
    quotes = [Quote(Side.BUY, D(990_000), D(10)), Quote(Side.BUY, D(900_000), D(10)),
              Quote(Side.SELL, D(1_010_000), D(10)), Quote(Side.SELL, D(1_020_000), D(10))]
    out = r.filter_quotes(quotes, make_book("999000", "1001000"))
    assert [q.price for q in out] == [D(990_000), D(1_010_000)]


def test_trading_windows_and_daily_move(store) -> None:  # type: ignore[no-untyped-def]
    # 2026-10-20 08:00 UTC = 11:30 Tehran (before the 11:40 open)
    now = [1792483200.0]
    r = RiskManager(RiskSettings(), store, clock=lambda: now[0])
    d = r.evaluate(make_book("1000000", "1001000", ts=now[0]), D(10**9))
    assert not d.allow_trading and d.code == "closed"
    now[0] += 3600  # 12:30 Tehran
    assert r.evaluate(make_book("1000000", "1001000", ts=now[0]), D(10**9)).allow_trading
    now[0] += 7200
    shock = r.evaluate(make_book("1045000", "1046000", ts=now[0]), D(10**9))
    assert not shock.allow_trading and shock.code == "daily_move"


def test_overnight_window_parsing() -> None:
    from datetime import time as t

    from nobitex_trader.risk import _in_window, _parse_window

    w = _parse_window("22:00-06:00")
    assert _in_window(t(23, 0), w) and _in_window(t(5, 0), w) and not _in_window(t(12, 0), w)


def test_daily_buy_cap(store) -> None:  # type: ignore[no-untyped-def]
    r = RiskManager(RiskSettings(max_daily_buy_base=D(100), trading_windows=[]), store)
    book = make_book("999000", "1001000")
    r.evaluate(book, D(10**9))
    r.record_fill(Fill("x", 1, Side.BUY, D(10**6), D(70), D(0), "usdt", time.time()))
    quotes = [Quote(Side.BUY, D(995_000), D(20)), Quote(Side.BUY, D(990_000), D(20)),
              Quote(Side.SELL, D(1_005_000), D(50))]
    out = r.filter_quotes(quotes, book)
    assert sum(q.amount for q in out if q.side is Side.BUY) <= D(30)
    assert any(q.side is Side.SELL for q in out)


def test_regime_filter_pauses_buys_in_downtrend(market) -> None:  # type: ignore[no-untyped-def]
    from nobitex_trader.domain.models import Candle

    candles = flat_candles(130)
    p0 = 1_000_000
    for i in range(20):  # clean, steady decline → efficiency ratio ≈ 1
        p = D(p0 - 4000 * (i + 1))
        candles.append(Candle(candles[-1].ts + 900, p + 2000, p + 2500, p - 500, p, D(10)))
    s = AdaptiveGridStrategy(GridSettings(sell_above_cost=False))
    inv = InventorySnapshot(D(250), D(250_000_000), None)
    d = s.decide(ctx(market, inv, book=make_book("919000", "921000"), candles=candles))
    assert d.info["regime"] == "trend"
    assert not any(q.side is Side.BUY for q in d.quotes)


def test_unstuck_lowers_sell_floor(market) -> None:  # type: ignore[no-untyped-def]
    s = AdaptiveGridStrategy(GridSettings(unstuck_after_hours=1, unstuck_max_loss_pct=D("0.5"),
                                          imbalance_weight=D(0)))
    inv = InventorySnapshot(D(840), D(150_000_000), avg_cost=D(1_050_000))
    t0 = time.time()
    first = s.decide(StrategyContext(market, make_book("999000", "1001000"), flat_candles(150),
                                     inv, t0))
    assert first.info["unstuck"] is False
    later = s.decide(StrategyContext(market, make_book("999000", "1001000"), flat_candles(150),
                                     inv, t0 + 3700))
    assert later.info["unstuck"] is True
    assert later.info["sell_floor"] < D(1_050_000)
