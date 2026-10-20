"""Bot-owned inventory and PnL accounting.

The bot manages a *virtual sub-account*: it starts from the configured capital
caps and only changes through its own fills. That way manual trades or other
funds in the same Nobitex account are never touched or mis-attributed.

PnL uses the average-cost method, with fees folded into cost basis (buys) or
proceeds (sells). Fee currency follows Nobitex: buy fees are charged in the base
currency received, sell fees in the quote currency received.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from nobitex_trader.domain.models import ZERO, Fill, Side
from nobitex_trader.persistence.store import Store
from nobitex_trader.strategy.base import InventorySnapshot

_STATE_KEY = "portfolio"


@dataclass(slots=True)
class PortfolioState:
    base: Decimal
    quote: Decimal
    avg_cost: Decimal | None
    realized_pnl: Decimal
    fees_quote: Decimal
    initial_base: Decimal
    initial_quote: Decimal
    initial_price: Decimal | None
    fill_count: int = 0


class Portfolio:
    def __init__(self, store: Store, base_ccy: str, quote_ccy: str) -> None:
        self._store = store
        self.base_ccy = base_ccy
        self.quote_ccy = quote_ccy
        self.state: PortfolioState | None = None

    def load(self) -> bool:
        raw = self._store.get(_STATE_KEY)
        if raw is None:
            return False
        self.state = PortfolioState(**raw)
        return True

    def initialize(self, base: Decimal, quote: Decimal, price: Decimal) -> None:
        self.state = PortfolioState(
            base=base,
            quote=quote,
            avg_cost=price if base > ZERO else None,
            realized_pnl=ZERO,
            fees_quote=ZERO,
            initial_base=base,
            initial_quote=quote,
            initial_price=price,
        )
        self._save()

    def _save(self) -> None:
        assert self.state is not None
        s = self.state
        self._store.set(
            _STATE_KEY,
            {
                "base": s.base,
                "quote": s.quote,
                "avg_cost": s.avg_cost,
                "realized_pnl": s.realized_pnl,
                "fees_quote": s.fees_quote,
                "initial_base": s.initial_base,
                "initial_quote": s.initial_quote,
                "initial_price": s.initial_price,
                "fill_count": s.fill_count,
            },
        )

    def apply_fill(self, fill: Fill) -> Decimal | None:
        """Update inventory; returns realized PnL (quote) for sells."""
        assert self.state is not None, "portfolio not initialized"
        s = self.state
        realized: Decimal | None = None
        fee_in_base = fill.fee_currency.lower() == self.base_ccy

        if fill.side is Side.BUY:
            received = fill.amount - (fill.fee if fee_in_base else ZERO)
            cost = fill.notional + (ZERO if fee_in_base else fill.fee)
            fee_quote = fill.fee * fill.price if fee_in_base else fill.fee
            new_base = s.base + received
            if new_base > ZERO:
                prev_cost = (s.avg_cost or fill.price) * s.base
                s.avg_cost = (prev_cost + cost) / new_base
            s.base = new_base
            s.quote -= fill.notional if fee_in_base else cost
        else:
            proceeds = fill.notional - (ZERO if fee_in_base else fill.fee)
            sold = fill.amount + (fill.fee if fee_in_base else ZERO)
            fee_quote = fill.fee * fill.price if fee_in_base else fill.fee
            basis = (s.avg_cost or fill.price) * sold
            realized = proceeds - basis
            s.realized_pnl += realized
            s.base -= sold
            s.quote += proceeds
            if s.base <= ZERO:
                s.base = max(s.base, ZERO)
                s.avg_cost = None

        s.fees_quote += fee_quote
        s.fill_count += 1
        self._save()
        return realized

    # -------------------------------------------------------------------- views

    def snapshot(self) -> InventorySnapshot:
        assert self.state is not None
        return InventorySnapshot(
            base=self.state.base, quote=self.state.quote, avg_cost=self.state.avg_cost
        )

    def equity(self, price: Decimal) -> Decimal:
        assert self.state is not None
        return self.state.quote + self.state.base * price

    def hold_equity(self, price: Decimal) -> Decimal:
        """Equity had the bot never traded (benchmark)."""
        assert self.state is not None
        return self.state.initial_quote + self.state.initial_base * price

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        assert self.state is not None
        if not self.state.avg_cost:
            return ZERO
        return (price - self.state.avg_cost) * self.state.base
