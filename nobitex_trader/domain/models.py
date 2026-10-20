"""Exchange-agnostic domain objects.

All monetary values are :class:`decimal.Decimal` — never ``float`` — because the
exchange works with exact decimal strings and rounding errors on order prices or
amounts lead to rejected orders or slow balance drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum

ZERO = Decimal(0)


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderStatus(StrEnum):
    NEW = "New"
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    DONE = "Done"
    CANCELED = "Canceled"
    REJECTED = "Rejected"

    @property
    def is_open(self) -> bool:
        return self in (OrderStatus.NEW, OrderStatus.ACTIVE, OrderStatus.INACTIVE)

    @classmethod
    def parse(cls, raw: str | None) -> OrderStatus:
        if not raw:
            return cls.NEW
        normalized = raw.strip().lower()
        for status in cls:
            if status.value.lower() == normalized:
                return status
        # Unknown statuses are treated as open so we keep tracking them rather than
        # silently forgetting an order that may still fill.
        return cls.ACTIVE


@dataclass(frozen=True, slots=True)
class OrderRequest:
    side: Side
    price: Decimal
    amount: Decimal
    client_order_id: str
    tag: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class Order:
    id: int | None
    client_order_id: str | None
    side: Side
    price: Decimal
    amount: Decimal
    matched: Decimal
    status: OrderStatus
    fee: Decimal = ZERO
    avg_price: Decimal | None = None
    created_at: float = field(default_factory=time.time)
    tag: str = ""

    @property
    def remaining(self) -> Decimal:
        rest = self.amount - self.matched
        return rest if rest > ZERO else ZERO

    @property
    def is_open(self) -> bool:
        return self.status.is_open

    @property
    def key(self) -> str:
        """Stable identity: prefer our own client id, fall back to exchange id."""
        return self.client_order_id or f"id:{self.id}"

    def with_updates(self, **changes: object) -> Order:
        return replace(self, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Fill:
    """An incremental execution of an order (delta, not cumulative)."""

    order_key: str
    order_id: int | None
    side: Side
    price: Decimal
    amount: Decimal
    fee: Decimal
    fee_currency: str
    timestamp: float
    tag: str = ""

    @property
    def notional(self) -> Decimal:
        return self.price * self.amount


@dataclass(frozen=True, slots=True)
class OrderBook:
    bids: tuple[tuple[Decimal, Decimal], ...]  # best (highest) first
    asks: tuple[tuple[Decimal, Decimal], ...]  # best (lowest) first
    last_trade_price: Decimal | None
    exchange_ts: float  # seconds
    received_ts: float  # local monotonic-ish wall clock, seconds

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return bid or ask or self.last_trade_price
        return (bid + ask) / 2

    @property
    def spread_ratio(self) -> Decimal | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None or bid <= ZERO:
            return None
        return (ask - bid) / bid

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.received_ts


@dataclass(frozen=True, slots=True)
class Candle:
    ts: int  # open time, unix seconds
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class PublicTrade:
    price: Decimal
    volume: Decimal
    side: Side
    ts: float


@dataclass(frozen=True, slots=True)
class Balance:
    currency: str
    total: Decimal
    blocked: Decimal = ZERO

    @property
    def free(self) -> Decimal:
        return self.total - self.blocked
