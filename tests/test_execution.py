from __future__ import annotations

from decimal import Decimal

import pytest

from nobitex_trader.backtest import aggregate, run_backtest, synthetic_candles
from nobitex_trader.domain.models import Fill, OrderStatus, PublicTrade, Side
from nobitex_trader.exchange.errors import OrderNotFound, UnknownOrderState
from nobitex_trader.exchange.paper import PaperVenue
from nobitex_trader.exchange.rate_limiter import SlidingWindowLimiter
from nobitex_trader.execution.order_manager import OrderManager
from nobitex_trader.strategy.base import Quote
from nobitex_trader.strategy.registry import create_strategy
from tests.conftest import make_book

D = Decimal


def make_oms(venue, market, store, fills):  # type: ignore[no-untyped-def]
    return OrderManager(venue, market, store, "nbxt", SlidingWindowLimiter(100, 60),
                        D("0.0015"), fills.append)


async def test_reconcile_places_keeps_and_cancels(market, store) -> None:  # type: ignore[no-untyped-def]
    venue = PaperVenue(market, {"usdt": D(100), "rls": D(10**9)})
    venue.on_book(make_book("999000", "1001000"))
    fills: list[Fill] = []
    oms = make_oms(venue, market, store, fills)

    q1 = [Quote(Side.BUY, D(990_000), D(10), "b1"), Quote(Side.SELL, D(1_010_000), D(10), "s1")]
    await oms.reconcile(q1, D(1_000_000))
    assert len(oms.open_orders) == 2
    ids = {o.id for o in oms.open_orders}

    # tiny price change inside tolerance → nothing replaced
    q2 = [Quote(Side.BUY, D(990_500), D(10), "b1"), Quote(Side.SELL, D(1_010_000), D(10), "s1")]
    await oms.reconcile(q2, D(1_000_000))
    assert {o.id for o in oms.open_orders} == ids

    # buy moves a lot, sell disappears → one cancel+place, one cancel
    q3 = [Quote(Side.BUY, D(980_000), D(10), "b1")]
    await oms.reconcile(q3, D(1_000_000))
    assert [o.price for o in oms.open_orders] == [D(980_000)]


async def test_fills_detected_as_deltas(market, store) -> None:  # type: ignore[no-untyped-def]
    venue = PaperVenue(market, {"usdt": D(0), "rls": D(10**9)})
    venue.on_book(make_book("999000", "1001000"))
    fills: list[Fill] = []
    oms = make_oms(venue, market, store, fills)
    await oms.reconcile([Quote(Side.BUY, D(990_000), D(10), "b1")], D(1_000_000))

    venue.on_trade(PublicTrade(D(990_000), D(4), Side.SELL, 0))  # partial
    await oms.sync()
    assert len(fills) == 1 and fills[0].amount == D(4) and fills[0].fee_currency == "usdt"
    venue.on_trade(PublicTrade(D(989_000), D(100), Side.SELL, 0))  # rest
    await oms.sync()
    assert [f.amount for f in fills] == [D(4), D(6)]
    assert oms.open_orders == []
    assert fills[1].fee == D(6) * market.maker_fee


async def test_unknown_placement_is_resolved_not_resent(market, store) -> None:  # type: ignore[no-untyped-def]
    venue = PaperVenue(market, {"usdt": D(0), "rls": D(10**9)})
    real_place = venue.place_order
    calls = []

    async def flaky_place(m, req):  # type: ignore[no-untyped-def]
        calls.append(req.client_order_id)
        await real_place(m, req)  # it DID reach the exchange…
        raise UnknownOrderState("timeout")  # …but we never saw the answer

    venue.place_order = flaky_place  # type: ignore[method-assign]
    oms = make_oms(venue, market, store, [])
    await oms.reconcile([Quote(Side.BUY, D(990_000), D(10), "b1")], D(1_000_000))
    assert len(oms.pending) == 1
    # next reconcile resolves by clientOrderId instead of placing a duplicate
    await oms.reconcile([Quote(Side.BUY, D(990_000), D(10), "b1")], D(1_000_000))
    assert len(calls) == 1 and not oms.pending
    assert len(await venue.open_orders(market)) == 1


async def test_recover_adopts_exchange_orders(market, store) -> None:  # type: ignore[no-untyped-def]
    venue = PaperVenue(market, {"usdt": D(50), "rls": D(10**9)})
    oms = make_oms(venue, market, store, [])
    await oms.reconcile([Quote(Side.SELL, D(1_010_000), D(10), "s1")], D(1_000_000))
    # a manual order (no prefix) must be ignored
    from nobitex_trader.domain.models import OrderRequest
    await venue.place_order(market, OrderRequest(Side.BUY, D(900_000), D(5), "manual1"))

    fresh = make_oms(venue, market, store, [])
    await fresh.recover()
    assert [o.client_order_id[:4] for o in fresh.open_orders] == ["nbxt"]


async def test_paper_venue_balance_checks(market) -> None:  # type: ignore[no-untyped-def]
    from nobitex_trader.domain.models import OrderRequest
    from nobitex_trader.exchange.errors import OrderRejected

    venue = PaperVenue(market, {"usdt": D(0), "rls": D(5_000_000)})
    with pytest.raises(OrderRejected):
        await venue.place_order(market, OrderRequest(Side.BUY, D(1_000_000), D(10), "a"))
    with pytest.raises(OrderRejected):
        await venue.place_order(market, OrderRequest(Side.BUY, D(1_000_000), D(1), "b"))  # small
    with pytest.raises(OrderNotFound):
        await venue.get_order(market, order_id=99)
    o = await venue.place_order(market, OrderRequest(Side.BUY, D(1_000_000), D(4), "c"))
    assert o.status is OrderStatus.ACTIVE
    bal = await venue.balances(["rls"])
    assert bal["rls"].blocked == D(4_000_000)


def test_aggregate_candles() -> None:
    candles = synthetic_candles(30)
    agg = aggregate(candles, 15)
    assert sum(1 for _ in agg) in (2, 3)
    assert agg[0].high == max(c.high for c in candles if c.ts - c.ts % 900 == agg[0].ts)


async def test_backtest_smoke(settings) -> None:  # type: ignore[no-untyped-def]
    candles = synthetic_candles(2 * 1440)
    result = await run_backtest(settings, create_strategy(settings), candles)
    assert result.fills > 0
    assert result.buys > 0 and result.sells > 0
    assert result.end_equity > 0
    # accounting identity: equity = quote + base * price
    assert result.end_equity == result.final_quote + result.final_base * result.end_price


async def test_low_budget_keeps_existing_orders(market, store) -> None:  # type: ignore[no-untyped-def]
    venue = PaperVenue(market, {"usdt": D(500), "rls": D(10**9)})
    venue.on_book(make_book("999000", "1001000"))
    limiter = SlidingWindowLimiter(4, 600)
    oms = OrderManager(venue, market, store, "nbxt", limiter, D("0.0015"), lambda f: None)
    first = [Quote(Side.BUY, D(990_000), D(10), "b1"), Quote(Side.SELL, D(1_010_000), D(10), "s1")]
    await oms.reconcile(first, D(1_000_000))
    # Budget nearly gone; a big move wants both orders replaced, only 2 placements left.
    moved = [Quote(Side.BUY, D(970_000), D(10), "b1"), Quote(Side.SELL, D(990_000), D(10), "s1"),
             Quote(Side.BUY, D(960_000), D(10), "b2"), Quote(Side.SELL, D(1_000_000), D(10), "s2")]
    await oms.reconcile(moved, D(980_000))
    assert len(oms.open_orders) >= 2  # never left with an empty book
    assert limiter.available() == 0


def test_far_levels_tolerate_more_drift(market, store) -> None:  # type: ignore[no-untyped-def]
    oms = make_oms(PaperVenue(market, {}), market, store, [])
    near = oms._tolerance_for(Quote(Side.BUY, D(995_000), D(10)), D(1_000_000))
    far = oms._tolerance_for(Quote(Side.BUY, D(970_000), D(10)), D(1_000_000))
    assert far > near * 3
