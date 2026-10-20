"""Candle-replay backtester.

It reuses the *production* components — strategy, risk manager, order manager,
paper venue and portfolio — with a simulated clock, so what you tune here is what
runs live. Each 1-minute candle is replayed as a price path
(open → low → high → close for green candles, open → high → low → close for red
ones) and resting orders fill when the path trades through them.

Fill realism (configurable under ``paper:``):

* ``trade_through_ticks`` — the path must trade *beyond* our price by N ticks
  before we count a fill (we are rarely first in the queue);
* ``volume_participation`` — at most this share of each candle's volume can fill
  our orders, so thin minutes give partial fills.

Reported metrics go beyond return: risk-adjusted ratios on daily equity, profit
factor of realized trades, and *markout* (adverse selection: how price moved
after our fills — persistently negative markout means we are being picked off).
"""

from __future__ import annotations

import bisect
import csv
import itertools
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from nobitex_trader.config import Settings
from nobitex_trader.domain.models import Candle, Fill, OrderBook, PublicTrade, Side
from nobitex_trader.exchange.paper import PaperVenue
from nobitex_trader.exchange.rate_limiter import SlidingWindowLimiter
from nobitex_trader.execution.order_manager import OrderManager
from nobitex_trader.persistence.store import Store
from nobitex_trader.portfolio import Portfolio
from nobitex_trader.risk import RiskManager
from nobitex_trader.strategy.base import Strategy, StrategyContext

HUGE = Decimal("1e12")
MARKOUT_MINUTES = (1, 5, 15, 60)


def resolution_minutes(resolution: str) -> int:
    r = resolution.upper()
    if r.endswith("D"):
        return int(r[:-1] or 1) * 1440
    return int(r)


def aggregate(candles: Sequence[Candle], minutes: int) -> list[Candle]:
    """Aggregate 1-minute candles into ``minutes`` buckets (last bucket may be partial)."""
    out: list[Candle] = []
    span = minutes * 60
    for c in candles:
        bucket = c.ts - c.ts % span
        if out and out[-1].ts == bucket:
            last = out[-1]
            out[-1] = Candle(bucket, last.open, max(last.high, c.high), min(last.low, c.low),
                             c.close, last.volume + c.volume)
        else:
            out.append(Candle(bucket, c.open, c.high, c.low, c.close, c.volume))
    return out


def load_csv(path: Path) -> list[Candle]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [
            Candle(int(r["ts"]), Decimal(r["open"]), Decimal(r["high"]), Decimal(r["low"]),
                   Decimal(r["close"]), Decimal(r.get("volume") or 0))
            for r in reader
        ]
    return sorted(rows, key=lambda c: c.ts)


def save_csv(path: Path, candles: Sequence[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])


@dataclass(slots=True)
class BacktestResult:
    start_equity: Decimal
    end_equity: Decimal
    hold_equity: Decimal
    start_price: Decimal
    end_price: Decimal
    fills: int
    buys: int
    sells: int
    realized_pnl: Decimal
    fees: Decimal
    max_drawdown_pct: Decimal
    final_base: Decimal
    final_quote: Decimal
    days: float
    orders_placed: int = 0
    orders_cancelled: int = 0
    sharpe: float = 0.0
    sortino: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    markout_bps: dict[int, float] = field(default_factory=dict)
    daily_returns: list[float] = field(default_factory=list)
    # hourly samples: (ts, equity, base, quote, price) — used by the walk-forward optimizer
    curve: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    fill_times: list[int] = field(default_factory=list)

    @property
    def return_pct(self) -> Decimal:
        return (self.end_equity / self.start_equity - 1) * 100

    @property
    def hold_return_pct(self) -> Decimal:
        return (self.hold_equity / self.start_equity - 1) * 100

    @property
    def inventory_pnl(self) -> Decimal:
        """Mark-to-market drift of held inventory (everything that is not realized)."""
        return self.end_equity - self.start_equity - self.realized_pnl

    @property
    def fill_ratio(self) -> float:
        return self.fills / self.orders_placed if self.orders_placed else 0.0

    @property
    def excess_return_pct(self) -> Decimal:
        """Return over simply holding the starting mix (the honest benchmark)."""
        return self.return_pct - self.hold_return_pct

    @property
    def usdt_return_pct(self) -> Decimal:
        """Return measured in USDT (does the bot beat simply holding dollars?)."""
        start = self.start_equity / self.start_price
        end = self.end_equity / self.end_price
        return (end / start - 1) * 100

    def summary(self) -> str:
        def toman(v: Decimal) -> str:
            return f"{int(v / 10):,}"

        markout = "  ".join(f"{m}m: {v:+.1f}bps" for m, v in self.markout_bps.items())
        per_10min = self.orders_placed / max(self.days, 1e-9) / 144
        return "\n".join([
            f"period              : {self.days:.1f} days",
            f"price               : {self.start_price} → {self.end_price} rial",
            f"start equity        : {toman(self.start_equity)} toman",
            f"end equity          : {toman(self.end_equity)} toman",
            f"strategy return     : {self.return_pct:.2f}% (rial)  |  "
            f"{self.usdt_return_pct:.2f}% (USDT)",
            f"buy & hold return   : {self.hold_return_pct:.2f}%",
            f"excess vs hold      : {self.excess_return_pct:+.2f}%",
            f"max drawdown        : {self.max_drawdown_pct:.2f}%",
            f"sharpe / sortino    : {self.sharpe:.2f} / {self.sortino:.2f} (daily, annualised)",
            f"profit factor       : {self.profit_factor:.2f}   win rate: {self.win_rate:.0%}",
            f"fills               : {self.fills} (buys {self.buys} / sells {self.sells})",
            f"markout (adverse)   : {markout or 'n/a'}",
            f"orders placed       : {self.orders_placed} (~{per_10min:.1f} per 10 min; cap 300)",
            f"fill ratio          : {self.fill_ratio:.0%} of placed orders filled",
            f"realized pnl        : {toman(self.realized_pnl)} toman (closed round trips)",
            f"inventory pnl       : {toman(self.inventory_pnl)} toman (mark-to-market drift)",
            f"fees paid           : {toman(self.fees)} toman",
            f"final inventory     : {self.final_base} USDT + {toman(self.final_quote)} toman",
        ])


class _SimClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ratios(daily: list[float]) -> tuple[float, float]:
    if len(daily) < 3:
        return 0.0, 0.0
    mean = sum(daily) / len(daily)
    var = sum((r - mean) ** 2 for r in daily) / (len(daily) - 1)
    dvar = sum(min(r, 0.0) ** 2 for r in daily) / len(daily)
    ann = math.sqrt(365)
    sharpe = mean / math.sqrt(var) * ann if var > 0 else 0.0
    sortino = mean / math.sqrt(dvar) * ann if dvar > 0 else (99.0 if mean > 0 else 0.0)
    return sharpe, sortino


async def run_backtest(
    settings: Settings,
    strategy: Strategy,
    candles_1m: Sequence[Candle],
    half_spread_pct: Decimal = Decimal("0.05"),
) -> BacktestResult:
    if len(candles_1m) < 2:
        raise ValueError("need at least 2 candles")
    market = settings.market.spec()
    clock = _SimClock()
    clock.now = float(candles_1m[0].ts)
    store = Store(":memory:")
    ps = settings.paper
    venue = PaperVenue(market, {market.src: ps.initial_base, market.dst: ps.initial_quote},
                       clock=clock, trade_through_ticks=ps.trade_through_ticks)
    portfolio = Portfolio(store, market.src, market.dst)
    portfolio.initialize(ps.initial_base, ps.initial_quote, candles_1m[0].open)
    risk = RiskManager(settings.risk, store, clock=clock)
    fills: list[tuple[Fill, Decimal | None]] = []

    def on_fill(fill: Fill) -> None:
        realized = portfolio.apply_fill(fill)
        risk.record_fill(fill)
        store.record_fill(fill, realized)
        strategy.on_fill(fill)
        fills.append((fill, realized))

    es = settings.engine
    oms = OrderManager(
        venue, market, store, es.client_order_prefix,
        SlidingWindowLimiter(es.order_rate_limit, es.order_rate_window_seconds, clock=clock),
        es.requote_tolerance_pct / 100, on_fill, clock=clock,
        filled_order_delay=settings.grid.filled_order_delay_seconds,
    )
    span = resolution_minutes(settings.grid.candle_resolution) * 60
    half = half_spread_pct / 100
    agg: list[Candle] = []
    prices: deque[tuple[float, Decimal]] = deque(maxlen=60)
    peak = portfolio.equity(candles_1m[0].open)
    max_dd = Decimal(0)
    day_equity: dict[int, Decimal] = {}
    curve: list[tuple[int, float, float, float, float]] = []

    def book_at(price: Decimal, ts: float) -> OrderBook:
        return OrderBook(
            bids=((price * (1 - half), HUGE),), asks=((price * (1 + half), HUGE),),
            last_trade_price=price, exchange_ts=ts, received_ts=ts,
        )

    for candle in candles_1m:
        clock.now = float(candle.ts)
        # 1) replay the candle path against resting orders
        venue.on_book(book_at(candle.open, clock.now))
        path = ((candle.low, candle.high) if candle.close >= candle.open
                else (candle.high, candle.low))
        volume = candle.volume * ps.volume_participation / 3 if candle.volume > 0 else HUGE
        for i, price in enumerate((*path, candle.close)):
            clock.now = candle.ts + 15.0 * (i + 1)
            venue.on_trade(PublicTrade(price, volume, Side.SELL, clock.now))
            prices.append((clock.now, price))
        await oms.sync()

        # 2) strategy candles (closed buckets + current partial bucket)
        bucket = candle.ts - candle.ts % span
        if agg and agg[-1].ts == bucket:
            last = agg[-1]
            agg[-1] = Candle(bucket, last.open, max(last.high, candle.high),
                             min(last.low, candle.low), candle.close, last.volume + candle.volume)
        else:
            agg.append(Candle(bucket, candle.open, candle.high, candle.low, candle.close,
                              candle.volume))
            if len(agg) > settings.grid.candle_history:
                del agg[0]

        # 3) decide at the close
        clock.now = candle.ts + 60.0
        book = book_at(candle.close, clock.now)
        equity = portfolio.equity(candle.close)
        day_equity[int(candle.ts // 86400)] = equity
        if not curve or candle.ts - curve[-1][0] >= 3600:
            snap = portfolio.snapshot()
            curve.append((candle.ts, float(equity), float(snap.base), float(snap.quote),
                          float(candle.close)))
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak * 100)
        decision = risk.evaluate(book, equity)
        if not decision.allow_trading:
            await oms.cancel_all()
            continue
        ctx = StrategyContext(market, book, list(agg), portfolio.snapshot(), clock.now,
                              price_history=tuple(prices))
        quotes = risk.filter_quotes(strategy.decide(ctx).quotes, book)
        assert book.mid is not None
        await oms.reconcile(quotes, book.mid)

    await oms.cancel_all()
    st = portfolio.state
    assert st is not None
    end_price = candles_1m[-1].close
    store.close()

    # ---- metrics -----------------------------------------------------------
    start_equity = st.initial_quote + st.initial_base * candles_1m[0].open
    equities = [start_equity] + [day_equity[d] for d in sorted(day_equity)]
    daily = [float(b / a - 1) for a, b in itertools.pairwise(equities) if a > 0]
    sharpe, sortino = _ratios(daily)
    pnls = [float(r) for _, r in fills if r is not None]
    gains = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p < 0)
    profit_factor = gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0)
    win_rate = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0.0

    times = [c.ts for c in candles_1m]
    markout: dict[int, float] = {}
    for minutes in MARKOUT_MINUTES:
        samples = []
        for fill, _ in fills:
            idx = bisect.bisect_left(times, fill.timestamp + minutes * 60)
            if idx >= len(candles_1m):
                continue
            move = (candles_1m[idx].close - fill.price) / fill.price
            samples.append(float(move if fill.side is Side.BUY else -move) * 10_000)
        if samples:
            markout[minutes] = sum(samples) / len(samples)

    return BacktestResult(
        start_equity=start_equity,
        end_equity=portfolio.equity(end_price),
        hold_equity=portfolio.hold_equity(end_price),
        start_price=candles_1m[0].open,
        end_price=end_price,
        fills=st.fill_count,
        buys=sum(1 for f, _ in fills if f.side is Side.BUY),
        sells=sum(1 for f, _ in fills if f.side is Side.SELL),
        realized_pnl=st.realized_pnl,
        fees=st.fees_quote,
        max_drawdown_pct=max_dd,
        final_base=st.base,
        final_quote=st.quote,
        days=(candles_1m[-1].ts - candles_1m[0].ts) / 86400,
        orders_placed=oms.placed_count,
        orders_cancelled=oms.cancel_count,
        sharpe=sharpe,
        sortino=sortino,
        profit_factor=profit_factor,
        win_rate=win_rate,
        markout_bps=markout,
        daily_returns=daily,
        curve=curve,
        fill_times=[int(f.timestamp) for f, _ in fills],
    )


def synthetic_candles(
    n: int, start_price: float = 1_000_000.0, seed: int = 7, drift: float = 0.0,
    vol: float = 0.0012, start_ts: int = 1_700_000_000,
) -> list[Candle]:
    """Mean-reverting random walk with drift — for tests and demos only."""
    import random

    rng = random.Random(seed)
    price = start_price
    anchor = start_price
    out: list[Candle] = []
    for i in range(n):
        anchor *= math.exp(drift)
        shock = rng.gauss(0, vol) + 0.02 * math.log(anchor / price)
        close = price * math.exp(shock)
        high = max(price, close) * (1 + abs(rng.gauss(0, vol / 2)))
        low = min(price, close) * (1 - abs(rng.gauss(0, vol / 2)))

        def d(x: float) -> Decimal:
            return Decimal(int(x))

        out.append(Candle(start_ts + 60 * i, d(price), d(high), d(low), d(close), Decimal(100)))
        price = close
    return out
