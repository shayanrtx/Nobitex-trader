"""Risk management: the layer allowed to say *no*.

Checks, in order:

1. **Manual halt** — persisted; requires ``nobitex-trader resume``.
2. **Stale data** — no fresh order book → pull all quotes.
3. **Abnormal spread** — thin/broken book → pull all quotes.
4. **Circuit breaker** — price moved more than X% inside a short window (news
   shock, depeg rumour, …) → pull quotes and cool down.
0. **Trading hours** — exchange-mandated windows (USDT/toman halts overnight).
   **Daily move pause** — price too far from the day's open (shock day).
5. **Daily loss limit** — equity fell X% since local-day start → pause until the
   next day (Asia/Tehran by default).
6. **Max drawdown** — equity fell X% from its peak → persistent halt.

It also sanitises quotes (price deviation, max open orders).
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from decimal import Decimal
from zoneinfo import ZoneInfo

from nobitex_trader.config import RiskSettings
from nobitex_trader.domain.models import Fill, OrderBook, Side
from nobitex_trader.persistence.store import Store
from nobitex_trader.strategy.base import Quote

HUNDRED = Decimal(100)
_STATE_KEY = "risk"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    allow_trading: bool
    reason: str = ""
    halted: bool = False  # persistent; needs manual resume
    new_event: bool = False  # first time this condition fired (for notifications)
    code: str = ""  # stable identifier of the condition (reason text may vary)


class RiskManager:
    def __init__(
        self, settings: RiskSettings, store: Store, clock: Callable[[], float] = time.time
    ) -> None:
        self.s = settings
        self._store = store
        self._clock = clock
        self._tz = ZoneInfo(settings.timezone)
        self._prices: deque[tuple[float, Decimal]] = deque()
        self._cooldown_until = 0.0
        self._last_code = ""
        state = store.get(_STATE_KEY) or {}
        self.halted_reason: str | None = state.get("halted_reason")
        self.peak_equity: Decimal | None = state.get("peak_equity")
        self.day: str | None = state.get("day")
        self.day_start_equity: Decimal | None = state.get("day_start_equity")
        self.day_paused: bool = bool(state.get("day_paused", False))
        self.price_day: str | None = state.get("price_day")
        self.day_open_price: Decimal | None = state.get("day_open_price")
        self.bought_today: Decimal = state.get("bought_today") or Decimal(0)
        self._windows = [_parse_window(w) for w in settings.trading_windows]

    def _save(self) -> None:
        self._store.set(
            _STATE_KEY,
            {
                "halted_reason": self.halted_reason,
                "peak_equity": self.peak_equity,
                "day": self.day,
                "day_start_equity": self.day_start_equity,
                "day_paused": self.day_paused,
                "price_day": self.price_day,
                "day_open_price": self.day_open_price,
                "bought_today": self.bought_today,
            },
        )

    # ------------------------------------------------------------------ controls

    def halt(self, reason: str) -> None:
        self.halted_reason = reason
        self._save()

    def resume(self) -> None:
        """Clear a manual/drawdown halt and reset the peak to current equity."""
        self.halted_reason = None
        self.peak_equity = None
        self.day_paused = False
        self._save()

    # ---------------------------------------------------------------- evaluation

    def evaluate(self, book: OrderBook | None, equity: Decimal | None) -> RiskDecision:
        now = self._clock()
        decision = self._evaluate(book, equity, now)
        new_event = (not decision.allow_trading) and decision.code != self._last_code
        self._last_code = decision.code if not decision.allow_trading else ""
        return RiskDecision(
            decision.allow_trading, decision.reason, decision.halted, new_event, decision.code
        )

    def _evaluate(self, book: OrderBook | None, equity: Decimal | None, now: float) -> RiskDecision:
        if self.halted_reason:
            return RiskDecision(False, f"halted: {self.halted_reason}", halted=True, code="halted")

        local = datetime.fromtimestamp(now, self._tz)
        if self._windows and not any(_in_window(local.time(), w) for w in self._windows):
            return RiskDecision(False, "outside trading hours", code="closed")

        if book is None or book.mid is None:
            return RiskDecision(False, "no order book", code="no_book")
        if book.age(now) > self.s.stale_data_seconds:
            return RiskDecision(False, f"stale market data ({book.age(now):.0f}s)", code="stale")
        spread = book.spread_ratio
        if spread is None:
            return RiskDecision(False, "one-sided order book", code="one_sided")
        if spread * HUNDRED > self.s.max_spread_pct:
            return RiskDecision(False, f"spread too wide ({spread * HUNDRED:.2f}%)", code="spread")

        mid = book.mid
        day = local.strftime("%Y-%m-%d")
        if day != self.price_day:
            self.price_day, self.day_open_price, self.bought_today = day, mid, Decimal(0)
            self._save()
        if self.s.daily_move_pause_pct > 0 and self.day_open_price:
            day_move = abs(mid / self.day_open_price - 1) * HUNDRED
            if day_move > self.s.daily_move_pause_pct:
                return RiskDecision(False, f"daily move {day_move:.2f}% beyond limit",
                                    code="daily_move")

        self._prices.append((now, mid))
        while self._prices and now - self._prices[0][0] > self.s.circuit_breaker_window_seconds:
            self._prices.popleft()
        lo = min(p for _, p in self._prices)
        hi = max(p for _, p in self._prices)
        move = (hi - lo) / lo * HUNDRED if lo > 0 else Decimal(0)
        if move > self.s.circuit_breaker_move_pct:
            self._cooldown_until = now + self.s.circuit_breaker_cooldown_seconds
            self._prices.clear()
            return RiskDecision(False, f"circuit breaker: {move:.2f}% move", code="circuit")
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            return RiskDecision(False, f"circuit breaker cooldown ({remaining:.0f}s)",
                                code="circuit")

        if equity is not None and equity > 0:
            verdict = self._check_equity(equity, now)
            if verdict is not None:
                return verdict
        return RiskDecision(True)

    def _check_equity(self, equity: Decimal, now: float) -> RiskDecision | None:
        day = datetime.fromtimestamp(now, self._tz).strftime("%Y-%m-%d")
        changed = False
        if day != self.day:
            self.day, self.day_start_equity, self.day_paused = day, equity, False
            changed = True
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
            changed = True

        result: RiskDecision | None = None
        assert self.peak_equity is not None and self.day_start_equity is not None
        drawdown = (self.peak_equity - equity) / self.peak_equity * HUNDRED
        daily = (self.day_start_equity - equity) / self.day_start_equity * HUNDRED
        if drawdown >= self.s.max_drawdown_pct:
            self.halted_reason = f"max drawdown {drawdown:.2f}% reached"
            changed = True
            result = RiskDecision(
                False, f"halted: {self.halted_reason}", halted=True, code="halted"
            )
        elif self.day_paused or daily >= self.s.max_daily_loss_pct:
            if not self.day_paused:
                self.day_paused = True
                changed = True
            result = RiskDecision(False, f"daily loss limit ({daily:.2f}%) — paused until tomorrow",
                                  code="daily_loss")
        if changed:
            self._save()
        return result

    # ------------------------------------------------------------------ quotes

    def record_fill(self, fill: Fill) -> None:
        """Track the regulatory daily buy cap."""
        if fill.side is Side.BUY:
            day = datetime.fromtimestamp(fill.timestamp, self._tz).strftime("%Y-%m-%d")
            if day != self.price_day:
                self.price_day, self.day_open_price, self.bought_today = day, None, Decimal(0)
            self.bought_today += fill.amount
            self._save()

    def filter_quotes(self, quotes: list[Quote], book: OrderBook) -> list[Quote]:
        mid = book.mid
        if mid is None:
            return []
        limit = self.s.max_price_deviation_pct / HUNDRED
        sane = [q for q in quotes if abs(q.price - mid) / mid <= limit]
        sane.sort(key=lambda q: abs(q.price - mid))
        if self.s.max_daily_buy_base > 0:
            # Resting buys count against the cap too, so never quote more than remains.
            room = self.s.max_daily_buy_base - self.bought_today
            capped = []
            for q in sane:
                if q.side is Side.BUY:
                    if q.amount > room:
                        continue
                    room -= q.amount
                capped.append(q)
            sane = capped
        return sane[: self.s.max_open_orders]


def _parse_window(text: str) -> tuple[dtime, dtime]:
    try:
        start, end = (dtime.fromisoformat(part.strip()) for part in text.split("-"))
    except ValueError as exc:
        raise ValueError(f"bad trading window {text!r}; expected HH:MM-HH:MM") from exc
    return start, end


def _in_window(t: dtime, window: tuple[dtime, dtime]) -> bool:
    start, end = window
    if start <= end:
        return start <= t < end
    return t >= start or t < end  # overnight window, e.g. 22:00-06:00
