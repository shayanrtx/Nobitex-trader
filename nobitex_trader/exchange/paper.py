"""Paper-trading venue: real market data, simulated order matching.

Resting limit orders fill when the live market trades through them:

* a buy at ``P`` fills when the best ask drops to ``≤ P`` or a public trade
  prints at ``≤ P``; a sell mirrors this;
* an order that would cross the book on arrival fills immediately at the touch
  as taker.

This is slightly optimistic about queue position (we assume we're first at our
price) and therefore a good *upper bound* — always validate in paper mode before
going live, and expect live results to be somewhat worse.

State (balances + orders) is persisted so paper sessions survive restarts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import (
    ZERO,
    Balance,
    Order,
    OrderBook,
    OrderRequest,
    OrderStatus,
    PublicTrade,
    Side,
)
from nobitex_trader.exchange.errors import OrderNotFound, OrderRejected
from nobitex_trader.persistence.store import Store

_STATE_KEY = "paper_venue"


class PaperVenue:
    def __init__(
        self,
        market: MarketSpec,
        initial_balances: dict[str, Decimal],
        store: Store | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._market = market
        self._store = store
        self._clock = clock
        self._orders: dict[int, Order] = {}
        self._balances: dict[str, Decimal] = {k.lower(): v for k, v in initial_balances.items()}
        self._next_id = 1
        self._book: OrderBook | None = None
        self._load()

    # ------------------------------------------------------------- persistence

    def _load(self) -> None:
        if self._store is None:
            return
        raw = self._store.get(_STATE_KEY)
        if not raw:
            return
        self._balances = {k: Decimal(v) for k, v in raw["balances"].items()}
        for o in raw["orders"]:
            order = Order(
                id=o["id"],
                client_order_id=o["cid"],
                side=Side(o["side"]),
                price=Decimal(o["price"]),
                amount=Decimal(o["amount"]),
                matched=Decimal(o["matched"]),
                status=OrderStatus(o["status"]),
                fee=Decimal(o["fee"]),
                avg_price=Decimal(o["avg"]) if o.get("avg") else None,
                created_at=o["created_at"],
            )
            self._orders[order.id] = order  # type: ignore[index]
        self._next_id = int(raw.get("next_id", 1))

    def _prune(self) -> None:
        """Keep open orders plus the most recent closed ones (bounded memory)."""
        if len(self._orders) <= 150:
            return
        closed = sorted(
            (o for o in self._orders.values() if not o.is_open), key=lambda o: o.created_at
        )[-50:]
        keep = [o for o in self._orders.values() if o.is_open] + closed
        self._orders = {o.id: o for o in keep}  # type: ignore[misc]

    def _save(self) -> None:
        self._prune()
        if self._store is None:
            return
        keep = list(self._orders.values())
        self._store.set(
            _STATE_KEY,
            {
                "balances": {k: str(v) for k, v in self._balances.items()},
                "next_id": self._next_id,
                "orders": [self._dump(o) for o in keep],
            },
        )

    @staticmethod
    def _dump(o: Order) -> dict[str, Any]:
        return {
            "id": o.id,
            "cid": o.client_order_id,
            "side": o.side.value,
            "price": str(o.price),
            "amount": str(o.amount),
            "matched": str(o.matched),
            "status": o.status.value,
            "fee": str(o.fee),
            "avg": str(o.avg_price) if o.avg_price else None,
            "created_at": o.created_at,
        }

    def _new_id(self) -> int:
        order_id = self._next_id
        self._next_id += 1
        return order_id

    # ------------------------------------------------------------ balance math

    def _bal(self, ccy: str) -> Decimal:
        return self._balances.get(ccy, ZERO)

    def _locked(self, ccy: str) -> Decimal:
        m = self._market
        total = ZERO
        for o in self._orders.values():
            if not o.is_open:
                continue
            if o.side is Side.BUY and ccy == m.dst:
                total += o.remaining * o.price
            elif o.side is Side.SELL and ccy == m.src:
                total += o.remaining
        return total

    # ------------------------------------------------------------ market input

    def on_book(self, book: OrderBook) -> None:
        self._book = book
        changed = False
        for order in list(self._orders.values()):
            if not order.is_open:
                continue
            if (order.side is Side.BUY and book.best_ask is not None
                    and book.best_ask <= order.price) or (
                    order.side is Side.SELL and book.best_bid is not None
                    and book.best_bid >= order.price):
                self._fill(order, order.remaining, order.price, maker=True)
                changed = True
        if changed:
            self._save()

    def on_trade(self, trade: PublicTrade) -> None:
        changed = False
        volume = trade.volume
        for order in sorted(self._orders.values(), key=lambda o: o.created_at):
            if not order.is_open or volume <= ZERO:
                continue
            crosses = (order.side is Side.BUY and trade.price <= order.price) or (
                order.side is Side.SELL and trade.price >= order.price
            )
            if crosses:
                qty = min(order.remaining, volume)
                self._fill(order, qty, order.price, maker=True)
                volume -= qty
                changed = True
        if changed:
            self._save()

    def _fill(self, order: Order, qty: Decimal, price: Decimal, maker: bool) -> None:
        m = self._market
        fee_rate = m.maker_fee if maker else m.taker_fee
        if order.side is Side.BUY:
            fee = qty * fee_rate  # in base
            self._balances[m.dst] = self._bal(m.dst) - qty * price
            self._balances[m.src] = self._bal(m.src) + qty - fee
        else:
            fee = qty * price * fee_rate  # in quote
            self._balances[m.src] = self._bal(m.src) - qty
            self._balances[m.dst] = self._bal(m.dst) + qty * price - fee
        matched = order.matched + qty
        prev_notional = (order.avg_price or ZERO) * order.matched
        avg = (prev_notional + qty * price) / matched
        status = OrderStatus.DONE if matched >= order.amount else OrderStatus.ACTIVE
        self._orders[order.id] = order.with_updates(  # type: ignore[index]
            matched=matched, fee=order.fee + fee, avg_price=avg, status=status
        )

    # ---------------------------------------------------------- TradingVenue API

    async def place_order(self, market: MarketSpec, request: OrderRequest) -> Order:
        if not market.meets_min_notional(request.price, request.amount):
            raise OrderRejected("SmallOrder: below minimum notional", "SmallOrder")
        if any(o.client_order_id == request.client_order_id and o.is_open
               for o in self._orders.values()):
            raise OrderRejected("DuplicateClientOrderId", "DuplicateClientOrderId")
        if request.side is Side.BUY:
            free = self._bal(market.dst) - self._locked(market.dst)
            if request.notional > free:
                raise OrderRejected("OverValueOrder: insufficient rial", "OverValueOrder")
        else:
            free = self._bal(market.src) - self._locked(market.src)
            if request.amount > free:
                raise OrderRejected("OverValueOrder: insufficient base", "OverValueOrder")

        order = Order(
            id=self._new_id(),
            client_order_id=request.client_order_id,
            side=request.side,
            price=request.price,
            amount=request.amount,
            matched=ZERO,
            status=OrderStatus.ACTIVE,
            created_at=self._clock(),
            tag=request.tag,
        )
        self._orders[order.id] = order  # type: ignore[index]
        book = self._book
        if book is not None:
            ask, bid = book.best_ask, book.best_bid
            if request.side is Side.BUY and ask is not None and ask <= request.price:
                self._fill(order, order.amount, ask, maker=False)
            elif request.side is Side.SELL and bid is not None and bid >= request.price:
                self._fill(order, order.amount, bid, maker=False)
        self._save()
        return self._orders[order.id]  # type: ignore[index]

    def _find(self, order_id: int | None, client_order_id: str | None) -> Order:
        if order_id is not None and order_id in self._orders:
            return self._orders[order_id]
        if client_order_id:
            for o in self._orders.values():
                if o.client_order_id == client_order_id:
                    return o
        raise OrderNotFound(f"paper order {order_id or client_order_id} not found")

    async def cancel_order(self, market: MarketSpec, order: Order) -> Order | None:
        current = self._find(order.id, order.client_order_id)
        if current.is_open:
            current = current.with_updates(status=OrderStatus.CANCELED)
            self._orders[current.id] = current  # type: ignore[index]
            self._save()
        return current

    async def get_order(
        self, market: MarketSpec, order_id: int | None = None, client_order_id: str | None = None
    ) -> Order:
        return self._find(order_id, client_order_id)

    async def open_orders(self, market: MarketSpec) -> list[Order]:
        return [o for o in self._orders.values() if o.is_open]

    async def balances(self, currencies: list[str]) -> dict[str, Balance]:
        return {
            c.lower(): Balance(c.lower(), self._bal(c.lower()), self._locked(c.lower()))
            for c in currencies
        }
