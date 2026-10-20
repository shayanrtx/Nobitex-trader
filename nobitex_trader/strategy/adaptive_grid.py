"""Adaptive, inventory-aware grid for range/swing trading (نوسان‌گیری).

How it works
------------
1. **Spacing from volatility** — grid step ``s`` = ``spacing_atr_mult × ATR / price``,
   clamped to ``[min_spacing_pct, max_spacing_pct]``. Calm market → tight grid, many
   small cycles; volatile market → wide grid, fewer but larger swings, and no
   getting run over by spikes.
2. **Trend-aware inventory target** — the share of equity held in base (USDT)
   targets ``target_base_ratio``, nudged up in an uptrend (fast EMA above slow EMA)
   and down in a downtrend, bounded by ``[min_base_ratio, max_base_ratio]``.
3. **Inventory skew** — the grid centre (reservation price) shifts away from mid in
   proportion to how far inventory is from target: holding too much USDT moves the
   sells closer and the buys further, and vice versa; order sizes are skewed too.
4. **Mean-reversion tilt** — when mid is stretched above the fair EMA the centre
   shifts down a little (sell the rip), and up when it is stretched below
   (buy the dip).
5. **Cost guard** — optionally, sells are never placed below
   ``avg_cost × (1 + fee + min_profit)``, so each cycle is profitable after fees.
6. **Maker only** — prices are clamped so they never cross the book.
7. **Order-book imbalance** — the centre leans toward the side with more resting
   size (a microprice-style short-term signal that reacts within seconds).
8. **Momentum guard** — after a sharp move inside a few minutes, quoting against
   the move is paused: no buying into a crash, no selling into a squeeze.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any

from nobitex_trader.config import GridSettings
from nobitex_trader.domain.models import Side
from nobitex_trader.strategy.base import Quote, Strategy, StrategyContext, StrategyDecision
from nobitex_trader.strategy.indicators import atr, clamp, ema

ONE = Decimal(1)
HUNDRED = Decimal(100)


class AdaptiveGridStrategy(Strategy):
    name = "adaptive_grid"

    def __init__(self, params: GridSettings, mean_reversion_weight: Decimal = Decimal("0.5")):
        self.p = params
        self.mean_reversion_weight = mean_reversion_weight
        self._cache: tuple[Any, tuple[Decimal, Decimal, Decimal, Decimal]] | None = None
        self._imbalance = Decimal(0)

    @property
    def warmup_candles(self) -> int:
        return max(self.p.fair_ema_period, self.p.trend_slow_ema, self.p.atr_period + 1)

    def decide(self, ctx: StrategyContext) -> StrategyDecision:
        p, market, book = self.p, ctx.market, ctx.book
        mid = book.mid
        if mid is None or mid <= 0 or len(ctx.candles) < self.warmup_candles:
            return StrategyDecision([], {"state": "warming_up", "candles": len(ctx.candles)})

        fair, fast, slow, atr_value = self._indicators(ctx)
        atr_ratio = atr_value / mid

        # 1. volatility-scaled spacing (fraction of price)
        # Floor: never tighter than the round-trip fee plus a 25% margin, whatever the
        # config says — the actual fee tier may be detected at runtime.
        min_s = max(p.min_spacing_pct / HUNDRED, market.round_trip_fee * Decimal("1.25"))
        s = clamp(p.spacing_atr_mult * atr_ratio, min_s, max(min_s, p.max_spacing_pct / HUNDRED))

        # 2. trend signal in [-1, 1], normalised by volatility
        vol_unit = max(atr_ratio, p.min_spacing_pct / HUNDRED)
        trend = clamp((fast - slow) / slow / (vol_unit * 2), -ONE, ONE) if slow > 0 else Decimal(0)
        target = clamp(
            p.target_base_ratio + p.trend_skew * trend, p.min_base_ratio, p.max_base_ratio
        )

        # 3. inventory deviation and skew
        inv = ctx.inventory
        base_ratio = inv.base_ratio(mid)
        deviation = base_ratio - target

        # 4. mean-reversion tilt, in units of spacing
        stretch = clamp((mid - fair) / fair / max(s, Decimal("1e-9")), -ONE * 2, ONE * 2)

        # 5. order-book imbalance in [-1, 1]: positive = bid pressure → lean upward
        # Smoothed (EMA over ticks): raw imbalance flickers with every book update and
        # would otherwise make the whole grid jitter.
        raw_imbalance = book_imbalance(book.bids, book.asks, p.imbalance_levels)
        self._imbalance += (raw_imbalance - self._imbalance) * Decimal("0.2")
        imbalance = self._imbalance

        shift = (
            p.inventory_skew * deviation * 2
            + self.mean_reversion_weight * stretch / 2
            - p.imbalance_weight * imbalance / 2
        ) * s
        # Snap the centre to a lattice of s/4 so it moves in discrete steps rather
        # than drifting with every tick (fewer cancel/replace cycles).
        lattice = max(fair * s / 4, market.price_tick)
        reservation = (mid * (ONE - shift) / lattice).to_integral_value() * lattice

        buy_mult = clamp(ONE - deviation * 2, Decimal(0), Decimal(2))
        sell_mult = clamp(ONE + deviation * 2, Decimal(0), Decimal(2))
        allow_buys = base_ratio < p.max_base_ratio
        allow_sells = base_ratio > p.min_base_ratio

        # 6. momentum guard: don't stand in front of a fast move
        momentum = short_term_return(ctx.price_history, ctx.now, p.momentum_window_seconds)
        guard = ""
        if momentum is not None:
            limit = p.momentum_threshold_spacing * s
            if momentum <= -limit:
                allow_buys, guard = False, "falling fast: buys paused"
            elif momentum >= limit:
                allow_sells, guard = False, "rising fast: sells paused"

        sell_floor: Decimal | None = None
        if p.sell_above_cost and inv.avg_cost:
            margin = market.maker_fee + p.min_profit_over_cost_pct / HUNDRED
            sell_floor = inv.avg_cost * (ONE + margin)

        quotes: list[Quote] = []
        best_bid, best_ask = book.best_bid, book.best_ask
        quote_budget, base_budget = inv.quote, inv.base

        offset = s / 2
        step = s
        seen: set[tuple[Side, Decimal]] = set()
        for level in range(1, p.levels + 1):
            if allow_buys:
                price = reservation * (ONE - offset)
                if best_ask is not None:
                    price = min(price, best_ask - market.price_tick)
                price = market.round_price(price, Side.BUY)
                amount = (
                    market.round_amount(p.order_notional * buy_mult / price)
                    if price > 0
                    else Decimal(0)
                )
                cost = price * amount
                if (
                    price > 0
                    and amount > 0
                    and market.meets_min_notional(price, amount)
                    and cost <= quote_budget
                    and (Side.BUY, price) not in seen
                ):
                    quotes.append(Quote(Side.BUY, price, amount, tag=f"b{level}"))
                    seen.add((Side.BUY, price))
                    quote_budget -= cost

            if allow_sells:
                price = reservation * (ONE + offset)
                if sell_floor is not None:
                    price = max(price, sell_floor)
                if best_bid is not None:
                    price = max(price, best_bid + market.price_tick)
                price = market.round_price(price, Side.SELL)
                amount = market.round_amount(
                    min(p.order_notional * sell_mult / price, base_budget)
                ) if price > 0 else Decimal(0)
                if (
                    amount > 0
                    and market.meets_min_notional(price, amount)
                    and (Side.SELL, price) not in seen
                ):
                    quotes.append(Quote(Side.SELL, price, amount, tag=f"s{level}"))
                    seen.add((Side.SELL, price))
                    base_budget -= amount

            offset += step
            step *= p.level_growth

        info = {
            "state": "quoting",
            "mid": mid,
            "fair": fair.quantize(market.price_tick),
            "atr_pct": (atr_ratio * HUNDRED).quantize(Decimal("0.001")),
            "spacing_pct": (s * HUNDRED).quantize(Decimal("0.001")),
            "trend": trend.quantize(Decimal("0.01")),
            "target_base_ratio": target.quantize(Decimal("0.001")),
            "base_ratio": base_ratio.quantize(Decimal("0.001")),
            "reservation": reservation.quantize(market.price_tick),
            "sell_floor": sell_floor.quantize(market.price_tick) if sell_floor else None,
            "imbalance": imbalance.quantize(Decimal("0.01")),
            "momentum_pct": (momentum * HUNDRED).quantize(Decimal("0.001"))
            if momentum is not None else None,
            "guard": guard or None,
        }
        return StrategyDecision(quotes, info)

    def _indicators(self, ctx: StrategyContext) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        """EMAs/ATR only change with candles — cache them between ticks."""
        last = ctx.candles[-1]
        key = (len(ctx.candles), last.ts, last.close, last.high, last.low)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        p = self.p
        closes = [c.close for c in ctx.candles]
        values = (
            ema(closes, p.fair_ema_period)[-1],
            ema(closes, p.trend_fast_ema)[-1],
            ema(closes, p.trend_slow_ema)[-1],
            atr(ctx.candles, p.atr_period) or Decimal(0),
        )
        self._cache = (key, values)
        return values


def book_imbalance(
    bids: Sequence[tuple[Decimal, Decimal]], asks: Sequence[tuple[Decimal, Decimal]], levels: int
) -> Decimal:
    """(bid notional − ask notional) / total over the top ``levels`` of each side."""
    bid_value = sum((p * q for p, q in bids[:levels]), Decimal(0))
    ask_value = sum((p * q for p, q in asks[:levels]), Decimal(0))
    total = bid_value + ask_value
    return (bid_value - ask_value) / total if total > 0 else Decimal(0)


def short_term_return(
    history: Sequence[tuple[float, Decimal]], now: float, window: float
) -> Decimal | None:
    """Price change over the last ``window`` seconds (fraction), or ``None``."""
    if not history:
        return None
    start = None
    for ts, price in history:
        if now - ts <= window:
            start = price
            break
    last = history[-1][1]
    if start is None or start <= 0 or history[0][0] > now - window * 0.5:
        return None  # not enough history yet
    return (last - start) / start
