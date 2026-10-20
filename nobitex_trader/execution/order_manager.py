"""Order lifecycle: reconcile desired quotes with live orders and detect fills.

Design notes
------------
* **Declarative** — each tick the strategy says which orders it *wants*; we diff
  that against what is resting and only cancel/place the difference. A
  tolerance band avoids churning orders on tiny price moves (orders keep queue
  priority and we stay far below the exchange's order rate limit).
* **Idempotent placement** — every order carries a unique ``clientOrderId``. If a
  placement times out we don't know whether it landed, so the order is kept as
  *pending* and resolved by ``clientOrderId`` on the next sync instead of being
  blindly re-sent (which could double the position).
* **Fills as deltas** — fills are derived from increases of ``matchedAmount``, so
  the same logic works with REST polling, websocket events, paper trading and
  restarts (state is persisted).
* **Only our orders** — orders without our client-id prefix (manual trades) are
  never touched.
"""

from __future__ import annotations

import logging
import secrets
import time
from collections.abc import Callable
from decimal import Decimal

from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import ZERO, Fill, Order, OrderRequest, OrderStatus, Side
from nobitex_trader.exchange.base import TradingVenue
from nobitex_trader.exchange.errors import (
    AuthError,
    ExchangeError,
    OrderNotFound,
    OrderRejected,
    TransientError,
    UnknownOrderState,
)
from nobitex_trader.exchange.rate_limiter import SlidingWindowLimiter
from nobitex_trader.fmt import SIDE_FA, fmt_rial
from nobitex_trader.persistence.store import Store
from nobitex_trader.strategy.base import Quote

log = logging.getLogger(__name__)

FillHandler = Callable[[Fill], None]


class OrderManager:
    def __init__(
        self,
        venue: TradingVenue,
        market: MarketSpec,
        store: Store,
        client_prefix: str,
        order_limiter: SlidingWindowLimiter,
        requote_tolerance: Decimal,
        on_fill: FillHandler,
        clock: Callable[[], float] = time.time,
        filled_order_delay: float = 0.0,
    ) -> None:
        if not client_prefix or len(client_prefix) > 8:
            raise ValueError("client_prefix must be 1..8 chars")
        self._venue = venue
        self._market = market
        self._store = store
        self._prefix = client_prefix
        self._limiter = order_limiter
        self._tolerance = requote_tolerance
        self._on_fill = on_fill
        self._clock = clock
        self.orders: dict[str, Order] = {}  # key -> latest known state (open or pending)
        self.pending: set[str] = set()  # placements with unknown outcome
        self.consecutive_rejections = 0
        self.placed_count = 0
        self.cancel_count = 0
        self._fill_delay = filled_order_delay
        self.market_closed_at = 0.0
        self._last_fill: dict[Side, float] = {}

    # ------------------------------------------------------------------ helpers

    def new_client_id(self, side: Side) -> str:
        # e.g. "nbxtb1729412345678a1b2" — unique, sortable, <= 32 chars
        return f"{self._prefix}{side.value[0]}{int(self._clock() * 1000)}{secrets.token_hex(2)}"

    def is_ours(self, order: Order) -> bool:
        return bool(order.client_order_id and order.client_order_id.startswith(self._prefix))

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self.orders.values() if o.is_open]

    def _track(self, order: Order) -> None:
        if order.is_open:
            self.orders[order.key] = order
        else:
            self.orders.pop(order.key, None)
        self._store.upsert_order(order)

    # ----------------------------------------------------------------- startup

    async def recover(self) -> None:
        """Rebuild state from the DB and the exchange after a (re)start."""
        persisted: dict[str, Order] = {}
        for row in self._store.open_order_rows():
            persisted[row["key"]] = Order(
                id=row["exchange_id"],
                client_order_id=row["client_order_id"],
                side=Side(row["side"]),
                price=Decimal(row["price"]),
                amount=Decimal(row["amount"]),
                matched=Decimal(row["matched"]),
                status=OrderStatus(row["status"]),
                fee=Decimal(row["fee"]),
                created_at=row["created_at"],
                tag=row["tag"] or "",
            )
        self.orders = dict(persisted)
        live = [o for o in await self._venue.open_orders(self._market) if self.is_ours(o)]
        for order in live:
            prev = self.orders.get(order.key)
            if prev is None:
                # Exchange knows an order of ours the DB doesn't: adopt with zero baseline
                # matched so any partial fill is accounted for.
                self.orders[order.key] = order.with_updates(matched=ZERO, fee=ZERO)
        await self.sync(live_open=live)
        log.info("recovered %d open order(s)", len(self.open_orders))

    # -------------------------------------------------------------------- sync

    async def sync(self, live_open: list[Order] | None = None) -> None:
        """Refresh tracked orders from the exchange and emit fills."""
        if live_open is None:
            live_open = [o for o in await self._venue.open_orders(self._market) if self.is_ours(o)]
        live_by_key = {o.key: o for o in live_open}

        for key, known in list(self.orders.items()):
            current = live_by_key.get(key)
            if current is None:
                # Not open anymore (filled / cancelled) or a pending placement: ask directly.
                try:
                    current = await self._venue.get_order(
                        self._market, order_id=known.id, client_order_id=known.client_order_id
                    )
                except OrderNotFound:
                    if key in self.pending:
                        log.info("pending order %s never reached the exchange", key)
                    else:
                        log.warning("tracked order %s unknown to the exchange; dropping", key)
                    self.pending.discard(key)
                    self.orders.pop(key, None)
                    self._store.upsert_order(known.with_updates(status=OrderStatus.CANCELED))
                    continue
                except TransientError as exc:
                    log.warning("sync: status of %s unavailable: %r", key, exc)
                    continue
            self.pending.discard(key)
            self._apply_update(known, current.with_updates(tag=known.tag or current.tag))

    def _apply_update(self, known: Order, current: Order) -> None:
        delta = current.matched - known.matched
        if delta > ZERO:
            fee_delta = max(current.fee - known.fee, ZERO)
            price = current.price
            if current.avg_price and known.avg_price and known.matched > ZERO:
                notional = current.avg_price * current.matched - known.avg_price * known.matched
                price = notional / delta if notional > ZERO else current.price
            elif current.avg_price:
                price = current.avg_price
            fill = Fill(
                order_key=current.key,
                order_id=current.id,
                side=current.side,
                price=price,
                amount=delta,
                fee=fee_delta,
                fee_currency=self._market.src if current.side is Side.BUY else self._market.dst,
                timestamp=self._clock(),
                tag=current.tag,
            )
            self._last_fill[fill.side] = fill.timestamp
            try:
                self._on_fill(fill)
            except Exception:
                log.exception("fill handler failed for %s", fill)
        self._track(current)

    # ---------------------------------------------------------------- reconcile

    def _tolerance_for(self, quote: Quote, mid: Decimal) -> Decimal:
        """Requote tolerance for one quote.

        * Far levels tolerate proportionally more drift (25% of their distance from
          mid): moving a level 3% away by 0.2% buys nothing but costs queue priority
          and order budget.
        * When the order budget runs low, tolerance widens so the bot keeps a stable
          book instead of burning the remaining budget on small adjustments.
        """
        distance = abs(quote.price - mid) / mid if mid > 0 else Decimal(0)
        available = self._limiter.available()
        if available < self._limiter.limit * 0.25:
            budget_factor = Decimal(3)
        elif available < self._limiter.limit * 0.5:
            budget_factor = Decimal(2)
        else:
            budget_factor = Decimal(1)
        return self._tolerance * budget_factor + distance * Decimal("0.25")

    def _matches(self, order: Order, quote: Quote, tolerance: Decimal) -> bool:
        if order.side is not quote.side:
            return False
        if abs(order.price - quote.price) / quote.price > tolerance:
            return False
        # Keep partially filled orders as long as a meaningful amount remains.
        return order.remaining >= quote.amount * Decimal("0.5")

    def plan(self, quotes: list[Quote], mid: Decimal) -> tuple[list[Order], list[Quote]]:
        """Return (orders to cancel, quotes to place), respecting the order budget."""
        unmatched = [o for o in self.open_orders if o.key not in self.pending]
        to_place: list[Quote] = []
        for quote in sorted(quotes, key=lambda q: abs(q.price - mid)):
            tolerance = self._tolerance_for(quote, mid)
            candidates = [o for o in unmatched if self._matches(o, quote, tolerance)]
            if candidates:
                best = min(candidates, key=lambda o: abs(o.price - quote.price))
                unmatched.remove(best)
            else:
                to_place.append(quote)

        # Budget: if we cannot place every replacement, keep the old order rather
        # than cancel it and leave a hole in the book. Orders with no wanted
        # counterpart on their side (e.g. paused by a guard) are always cancelled.
        # Fill delay: no *new* orders on a side that just traded (existing ones stay).
        now = self._clock()
        cooling = {side for side, ts in self._last_fill.items() if now - ts < self._fill_delay}
        deferred = [q for q in to_place if q.side in cooling]
        to_place = [q for q in to_place if q.side not in cooling]
        available = max(self._limiter.available(), 0)
        deferred += to_place[available:]
        to_place = to_place[:available]
        for quote in deferred:
            same_side = [o for o in unmatched if o.side is quote.side]
            if same_side:
                unmatched.remove(min(same_side, key=lambda o: abs(o.price - quote.price)))
        if deferred and not cooling:
            log.warning("order budget low: %d replacement(s) deferred", len(deferred))
        return unmatched, to_place

    async def reconcile(self, quotes: list[Quote], mid: Decimal) -> None:
        if self.pending:
            # Don't stack new orders on top of placements in an unknown state.
            await self.sync()
            if self.pending:
                return
        to_cancel, to_place = self.plan(quotes, mid)
        for order in to_cancel:
            await self.cancel(order)
        for quote in to_place:
            if not self._limiter.try_acquire():
                break
            await self.place(quote)

    async def place(self, quote: Quote) -> Order | None:
        request = OrderRequest(
            side=quote.side,
            price=quote.price,
            amount=quote.amount,
            client_order_id=self.new_client_id(quote.side),
            tag=quote.tag,
        )
        placeholder = Order(
            id=None,
            client_order_id=request.client_order_id,
            side=quote.side,
            price=quote.price,
            amount=quote.amount,
            matched=ZERO,
            status=OrderStatus.NEW,
            created_at=self._clock(),
            tag=quote.tag,
        )
        try:
            order = await self._venue.place_order(self._market, request)
        except UnknownOrderState as exc:
            log.warning("placement outcome unknown for %s: %r", request.client_order_id, exc)
            self.pending.add(placeholder.key)
            self._track(placeholder)
            return None
        except OrderRejected as exc:
            if exc.code == "MarketClosed":
                # Not a fault of ours: the market is halted (e.g. overnight hours).
                self.market_closed_at = self._clock()
                log.info("market closed; order not placed")
                return None
            self.consecutive_rejections += 1
            log.warning("order rejected (%s): %s %s @ %s", exc.code, quote.side.value,
                        quote.amount, quote.price)
            self._store.log_event(
                "order",
                f"رد سفارش {SIDE_FA[quote.side.value]} {quote.amount} تتر به قیمت "
                f"{fmt_rial(quote.price)}: {exc.code}",
                level="warning",
            )
            return None
        except AuthError:
            raise
        except ExchangeError as exc:
            log.warning("order placement failed: %r", exc)
            return None
        self.consecutive_rejections = 0
        self.placed_count += 1
        if order.is_open:
            order = order.with_updates(tag=quote.tag, matched=ZERO, fee=ZERO)
        log.info("placed %s %s @ %s [%s]", order.side.value, order.amount, order.price, quote.tag)
        self._store.log_event(
            "order",
            f"ثبت {SIDE_FA[order.side.value]} {order.amount} تتر به قیمت {fmt_rial(order.price)}"
            f" (پله {quote.tag[1:]})",
            data={"id": order.id, "cid": order.client_order_id},
        )
        if order.is_open:
            self._track(order)
        else:
            # Filled (or rejected) immediately: run through the fill path.
            self._apply_update(placeholder, order.with_updates(tag=quote.tag))
        return order

    async def cancel(self, order: Order) -> None:
        self.cancel_count += 1
        try:
            final = await self._venue.cancel_order(self._market, order)
        except OrderNotFound:
            final = None
        except AuthError:
            raise
        except ExchangeError as exc:
            log.warning("cancel %s failed: %r", order.key, exc)
            return
        self._store.log_event(
            "order",
            f"لغو {SIDE_FA[order.side.value]} {order.remaining} تتر به قیمت {fmt_rial(order.price)}"
            f" (پله {order.tag[1:]})",
            data={"id": order.id},
        )
        if final is not None:
            self._apply_update(order, final.with_updates(tag=order.tag))
        else:
            # Resolve the real final state (it may have filled meanwhile).
            try:
                current = await self._venue.get_order(
                    self._market, order_id=order.id, client_order_id=order.client_order_id
                )
                self._apply_update(order, current.with_updates(tag=order.tag))
            except ExchangeError:
                log.warning("could not confirm cancel of %s; will resync", order.key)

    async def cancel_all(self) -> None:
        for order in list(self.open_orders):
            await self.cancel(order)
