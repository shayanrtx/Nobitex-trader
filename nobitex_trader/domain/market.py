"""Market specification and exchange precision rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR, Decimal

from nobitex_trader.domain.models import Side


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Static description of a single spot market.

    Prices on IRT markets are expressed in **rials** (not toman), and ``src``/``dst``
    are the lowercase currency codes the order API expects (e.g. ``usdt``/``rls``).
    """

    symbol: str  # e.g. "USDTIRT" — used by market-data endpoints & websocket channels
    src: str  # base currency for orders, e.g. "usdt"
    dst: str  # quote currency for orders, e.g. "rls"
    price_tick: Decimal
    amount_step: Decimal
    min_notional: Decimal
    maker_fee: Decimal
    taker_fee: Decimal

    # -- rounding ---------------------------------------------------------------

    def round_price(self, price: Decimal, side: Side) -> Decimal:
        """Round to the tick *away from the market* so we never cross unintentionally.

        Buys round down, sells round up.
        """
        rounding = ROUND_FLOOR if side is Side.BUY else ROUND_CEILING
        return _quantize(price, self.price_tick, rounding)

    def round_amount(self, amount: Decimal) -> Decimal:
        """Amounts always round down so we never ask for more than we can afford."""
        return _quantize(amount, self.amount_step, ROUND_DOWN)

    def meets_min_notional(self, price: Decimal, amount: Decimal) -> bool:
        return price * amount >= self.min_notional

    @property
    def round_trip_fee(self) -> Decimal:
        """Fee cost of one maker buy + one maker sell, as a ratio."""
        return self.maker_fee * 2


def _quantize(value: Decimal, step: Decimal, rounding: str) -> Decimal:
    if step <= 0:
        raise ValueError("step must be positive")
    units = (value / step).to_integral_value(rounding=rounding)
    result = units * step
    # Normalise exponent to the step's so string formatting is clean ("1230" not "1.23E+3").
    return result.quantize(step)
