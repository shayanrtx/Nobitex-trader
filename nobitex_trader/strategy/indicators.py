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
