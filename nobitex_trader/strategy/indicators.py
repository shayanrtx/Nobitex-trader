"""Technical indicators on Decimal series (pure functions, no state)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from decimal import Decimal

from nobitex_trader.domain.models import Candle


def ema(values: Sequence[Decimal], period: int) -> list[Decimal]:
    """Exponential moving average seeded with the SMA of the first ``period`` values."""
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period:
        return []
    alpha = Decimal(2) / Decimal(period + 1)
    seed = sum(values[:period], Decimal(0)) / period
    out = [seed]
    for value in values[period:]:
        out.append(out[-1] + alpha * (value - out[-1]))
    return out


def true_ranges(candles: Sequence[Candle]) -> list[Decimal]:
    out: list[Decimal] = []
    for prev, cur in itertools.pairwise(candles):
        out.append(
            max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        )
    return out


def atr(candles: Sequence[Candle], period: int) -> Decimal | None:
    """Wilder's Average True Range; ``None`` until enough data."""
    trs = true_ranges(candles)
    if len(trs) < period:
        return None
    value = sum(trs[:period], Decimal(0)) / period
    for tr in trs[period:]:
        value = (value * (period - 1) + tr) / period
    return value


def clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


def efficiency_ratio(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Kaufman efficiency ratio: |net move| / sum of |moves| over ``period`` steps.

    ~0 = choppy/ranging (ideal for a grid), ~1 = clean trend.
    """
    if period <= 0 or len(values) <= period:
        return None
    window = values[-(period + 1):]
    path = sum((abs(b - a) for a, b in itertools.pairwise(window)), Decimal(0))
    if path == 0:
        return Decimal(0)
    return abs(window[-1] - window[0]) / path
