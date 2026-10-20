"""The trading loop: market data → risk → strategy → order reconciliation.

The loop is event-driven: it wakes on every order-book update (websocket pushes
arrive many times per second on USDT-IRT) and on private order events, with a
timer as a fallback. Requotes are rate-limited by ``min_requote_seconds`` and
the order manager's tolerance band, so reacting fast never means spamming the
exchange.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Any

from nobitex_trader.config import Settings
from nobitex_trader.data.market_data import MarketDataFeed
from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import Fill, OrderBook
from nobitex_trader.exchange.base import TradingVenue
from nobitex_trader.exchange.errors import AuthError, TransientError
from nobitex_trader.exchange.rate_limiter import SlidingWindowLimiter
from nobitex_trader.execution.order_manager import OrderManager
from nobitex_trader.fmt import GUARD_FA, RISK_FA, SIDE_FA, fmt_rial
from nobitex_trader.notify import Notifier
from nobitex_trader.persistence.store import Store
from nobitex_trader.portfolio import Portfolio
from nobitex_trader.risk import RiskManager
from nobitex_trader.strategy.base import InventorySnapshot, Strategy, StrategyContext

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 20
MAX_CONSECUTIVE_REJECTIONS = 10
PRICE_HISTORY_SECONDS = 900
_CONTROL_KEY = "control"


class TradingEngine:
    def __init__(
        self,
        settings: Settings,
        market: MarketSpec,
        feed: MarketDataFeed,
        venue: TradingVenue,
        strategy: Strategy,
        store: Store,
        notifier: Notifier,
    ) -> None:
        self.settings = settings
        self.market = market
        self.feed = feed
        self.venue = venue
        self.strategy = strategy
        self.store = store
        self.notifier = notifier
        self.portfolio = Portfolio(store, market.src, market.dst)
        self.risk = RiskManager(settings.risk, store)
        es = settings.engine
        self.oms = OrderManager(
            venue=venue,
            market=market,
            store=store,
            client_prefix=es.client_order_prefix,
            order_limiter=SlidingWindowLimiter(es.order_rate_limit, es.order_rate_window_seconds),
            requote_tolerance=es.requote_tolerance_pct / 100,
            on_fill=self._on_fill,
        )
        self._stop = asyncio.Event()
        self._order_event = asyncio.Event()
        self._cancel_all_requested = False
        self._exchange_caps: tuple[Decimal, Decimal] | None = None
        self._price_history: deque[tuple[float, Decimal]] = deque()
        self.paused: bool = bool((store.get(_CONTROL_KEY) or {}).get("paused", False))
        self.started_at = time.time()
        self.last_info: dict[str, Any] = {}
        self.last_risk_reason = ""
        self._risk_code = ""
        feed.on_book(self._record_price)

    # ---------------------------------------------------------------- controls

    def stop(self) -> None:
        self._stop.set()

    def notify_order_event(self) -> None:
        """Called by the private websocket channel: triggers an immediate sync."""
        self._order_event.set()
        self.feed.updated.set()

    def pause(self, source: str = "داشبورد") -> None:
        self.paused = True
        self._cancel_all_requested = True
        self.store.set(_CONTROL_KEY, {"paused": True})
        self.store.log_event("control", f"توقف موقت توسط {source}", level="warning")
        self.notifier.send(f"⏸ ربات متوقف موقت شد ({source})")
        self.feed.updated.set()

    def resume(self, source: str = "داشبورد") -> None:
        self.paused = False
        self.risk.resume()
        self.oms.consecutive_rejections = 0
        self.store.set(_CONTROL_KEY, {"paused": False})
        self.store.log_event("control", f"ادامه‌ی معاملات توسط {source}")
        self.notifier.send(f"▶️ ربات ادامه داد ({source})")
        self.feed.updated.set()

    def request_cancel_all(self, source: str = "داشبورد") -> None:
        self._cancel_all_requested = True
        self.store.log_event("control", f"درخواست لغو همه‌ی سفارش‌ها توسط {source}", level="warning")
        self.feed.updated.set()

    # ------------------------------------------------------------------ status

    def status(self) -> dict[str, Any]:
        """Snapshot for the dashboard."""
        book = self.feed.orderbook
        mid = book.mid if book else None
        st = self.portfolio.state
        out: dict[str, Any] = {
            "mode": self.settings.mode,
            "market": self.market.symbol,
            "strategy": self.strategy.name,
            "maker_fee_pct": self.market.maker_fee * 100,
            "paused": self.paused,
            "halted": self.risk.halted_reason,
            "risk_reason": self.last_risk_reason,
            "risk_code": self._risk_code,
            "uptime_seconds": int(time.time() - self.started_at),
            "best_bid": book.best_bid if book else None,
            "best_ask": book.best_ask if book else None,
            "mid": mid,
            "book_age_seconds": round(book.age(), 1) if book else None,
            "open_orders": len(self.oms.open_orders),
            "strategy_info": self.last_info,
        }
        if st is not None and mid is not None:
            equity = self.portfolio.equity(mid)
            start = st.initial_quote + st.initial_base * (st.initial_price or mid)
            hold = self.portfolio.hold_equity(mid)
            out.update({
                "base": st.base,
                "quote": st.quote,
                "avg_cost": st.avg_cost,
                "equity": equity,
                "equity_usdt": equity / mid,
                "start_equity": start,
                "hold_equity": hold,
                "return_pct": (equity / start - 1) * 100 if start else None,
                "hold_return_pct": (hold / start - 1) * 100 if start else None,
                "realized_pnl": st.realized_pnl,
                "unrealized_pnl": self.portfolio.unrealized_pnl(mid),
                "fees": st.fees_quote,
                "fills": st.fill_count,
                "day_start_equity": self.risk.day_start_equity,
                "peak_equity": self.risk.peak_equity,
            })
            if self.risk.day_start_equity:
                out["today_pnl"] = equity - self.risk.day_start_equity
        return out

    # ----------------------------------------------------------------- startup

    def _record_price(self, book: OrderBook) -> None:
        mid = book.mid
        if mid is None:
            return
        now = book.received_ts
        if self._price_history and now - self._price_history[-1][0] < 1.0:
            self._price_history[-1] = (self._price_history[-1][0], mid)
        else:
            self._price_history.append((now, mid))
        while self._price_history and now - self._price_history[0][0] > PRICE_HISTORY_SECONDS:
            self._price_history.popleft()

    async def _init_portfolio(self) -> None:
        mid = self.feed.mid
        if mid is None:
            raise RuntimeError("no market price available")
        if self.portfolio.load():
            log.info("loaded portfolio state")
            return
        if self.settings.mode == "paper":
            base, quote = self.settings.paper.initial_base, self.settings.paper.initial_quote
        else:
            balances = await self.venue.balances([self.market.src, self.market.dst])
            base = min(self.settings.capital.max_base, balances[self.market.src].free)
            quote = min(self.settings.capital.max_quote, balances[self.market.dst].free)
        self.portfolio.initialize(base, quote, mid)
        self.store.log_event(
            "portfolio",
            f"شروع حساب ربات: {base} تتر و {fmt_rial(quote)} — قیمت {fmt_rial(mid)}",
        )
        log.info("initialized portfolio base=%s quote=%s @ %s", base, quote, mid)

    async def _refresh_exchange_caps(self) -> None:
        """Live safety: never budget more than the wallet actually holds."""
        if self.settings.mode != "live":
            return
        try:
            b = await self.venue.balances([self.market.src, self.market.dst])
        except TransientError as exc:
            log.warning("balance check failed: %r", exc)
            return
        caps = (b[self.market.src].total, b[self.market.dst].total)
        st = self.portfolio.state
        assert st is not None
        if caps[0] < st.base * Decimal("0.99") or caps[1] < st.quote * Decimal("0.99"):
            log.warning("wallet below bot book balance (USDT %s / rial %s)", caps[0], caps[1])
            self.store.log_event(
                "risk",
                f"موجودی کیف پول ({caps[0]} تتر / {fmt_rial(caps[1])}) کمتر از موجودی ربات است؛ "
                "بودجه محدود شد",
                level="warning",
            )
            self.notifier.send(
                "⚠️ موجودی کیف پول کمتر از موجودی دفتری ربات است "
                f"(USDT {caps[0]} / ریال {caps[1]}). بودجه محدود شد."
            )
        self._exchange_caps = caps

    def _inventory(self) -> InventorySnapshot:
        snap = self.portfolio.snapshot()
        if self._exchange_caps is None:
            return snap
        return InventorySnapshot(
            base=min(snap.base, self._exchange_caps[0]),
            quote=min(snap.quote, self._exchange_caps[1]),
            avg_cost=snap.avg_cost,
        )

    # -------------------------------------------------------------------- fills

    def _on_fill(self, fill: Fill) -> None:
        realized = self.portfolio.apply_fill(fill)
        self.store.record_fill(fill, realized)
        self.strategy.on_fill(fill)
        log.info("FILL %s %s @ %s fee=%s %s pnl=%s", fill.side.value, fill.amount, fill.price,
                 fill.fee, fill.fee_currency, realized)
        journal = f"{SIDE_FA[fill.side.value]} {fill.amount} تتر به قیمت {fmt_rial(fill.price)}"
        if realized is not None:
            journal += f" — سود/زیان: {fmt_rial(realized)}"
        self.store.log_event("fill", journal, data={"order": fill.order_key, "tag": fill.tag,
                                                    "fee": fill.fee, "fee_ccy": fill.fee_currency})
        if self.settings.notify.notify_fills:
            side = "🟢 خرید" if fill.side.value == "buy" else "🔴 فروش"
            msg = f"{side} {fill.amount} USDT @ {fmt_rial(fill.price)}"
            if realized is not None:
                msg += f"\nسود/زیان تحقق‌یافته: {fmt_rial(realized)}"
            self.notifier.send(msg)
        # React immediately: re-place the grid around the new inventory.
        self.feed.updated.set()

    # --------------------------------------------------------------------- loop

    async def run(self) -> None:
        es = self.settings.engine
        await self._init_portfolio()
        await self._refresh_exchange_caps()
        await self.oms.recover()
        mode_fa = "واقعی" if self.settings.mode == "live" else "آزمایشی"
        self.store.log_event("lifecycle", f"ربات شروع به کار کرد (حالت {mode_fa})")
        self.notifier.send(
            f"🤖 ربات شروع شد — حالت {self.settings.mode} — بازار {self.market.symbol}"
        )

        last_sync = last_quote = last_equity = last_caps = 0.0
        last_prune = time.time()
        errors = 0
        try:
            while not self._stop.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.feed.updated.wait(), timeout=es.tick_seconds)
                self.feed.updated.clear()
                if self._stop.is_set():
                    break
                now = time.time()
                try:
                    if self._cancel_all_requested:
                        self._cancel_all_requested = False
                        await self.oms.cancel_all()
                    if self._order_event.is_set() or now - last_sync >= es.order_sync_seconds:
                        self._order_event.clear()
                        await self.oms.sync()
                        last_sync = now
                    if now - last_caps >= 600:
                        await self._refresh_exchange_caps()
                        last_caps = now
                    if now - last_quote >= es.min_requote_seconds:
                        await self._step()
                        last_quote = now
                    if now - last_equity >= es.equity_snapshot_seconds:
                        self._snapshot_equity(now)
                        last_equity = now
                    if now - last_prune >= 86400:
                        self.store.prune()
                        last_prune = now
                    self._heartbeat()
                    errors = 0
                except AuthError as exc:
                    self.store.log_event(
                        "error", f"خطای احراز هویت — ربات متوقف شد: {exc}", level="error"
                    )
                    self.notifier.send(f"⛔️ خطای احراز هویت — ربات متوقف شد: {exc}")
                    raise
                except TransientError as exc:
                    errors += 1
                    log.warning("transient error (%d): %r", errors, exc)
                    if errors in (1, 5):
                        self.store.log_event("error", f"خطای موقت ارتباط: {exc!r}", level="warning")
                    await asyncio.sleep(min(2 * errors, 30))
                except Exception as exc:
                    errors += 1
                    log.exception("unexpected error (%d)", errors)
                    self.store.log_event("error", f"خطای غیرمنتظره: {exc!r}", level="error")
                    if errors in (1, 5) or errors % 10 == 0:
                        self.notifier.send(f"⚠️ خطای غیرمنتظره: {exc!r}")
                    await asyncio.sleep(min(2 * errors, 30))
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    self.notifier.send("⛔️ خطاهای پیاپی زیاد — ربات متوقف شد")
                    raise RuntimeError("too many consecutive errors")
        finally:
            await self._shutdown()

    async def _step(self) -> None:
        if self.paused:
            if self.oms.open_orders:
                await self.oms.cancel_all()
            self.last_risk_reason = "paused"
            self._risk_code = "paused"
            return
        book = self.feed.orderbook
        mid = book.mid if book else None
        equity = self.portfolio.equity(mid) if mid else None
        decision = self.risk.evaluate(book, equity)
        if not decision.allow_trading:
            self.last_risk_reason = decision.reason
            self._risk_code = decision.code
            if self.oms.open_orders:
                log.warning("risk: %s — cancelling all orders", decision.reason)
                await self.oms.cancel_all()
            if decision.new_event:
                self.store.log_event(
                    "risk", RISK_FA.get(decision.code, decision.reason), level="warning",
                    data={"detail": decision.reason},
                )
                self.notifier.send(f"🛑 توقف معاملات: {decision.reason}")
            return
        if self._risk_code:
            previous = RISK_FA.get(self._risk_code, self._risk_code)
            self.store.log_event("risk", f"معاملات از سر گرفته شد (پس از: {previous})")
            self._risk_code = ""
        self.last_risk_reason = ""
        if self.oms.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
            reason = f"{self.oms.consecutive_rejections} consecutive order rejections"
            self.risk.halt(reason)
            self.store.log_event(
                "risk", f"توقف کامل: {self.oms.consecutive_rejections} سفارش پیاپی رد شد",
                level="error",
            )
            self.notifier.send("⛔️ سفارش‌ها پیاپی رد شدند — ربات متوقف شد (resume لازم است)")
            await self.oms.cancel_all()
            return

        assert book is not None and mid is not None
        ctx = StrategyContext(
            market=self.market,
            book=book,
            candles=self.feed.candles.snapshot(),
            inventory=self._inventory(),
            now=time.time(),
            price_history=tuple(self._price_history),
        )
        result = self.strategy.decide(ctx)
        prev_guard = self.last_info.get("guard")
        self.last_info = result.info
        if result.info.get("guard") != prev_guard and result.info.get("guard"):
            guard = str(result.info["guard"])
            self.store.log_event("strategy", GUARD_FA.get(guard, guard), level="warning")
        quotes = self.risk.filter_quotes(result.quotes, book)
        await self.oms.reconcile(quotes, mid)

    def _snapshot_equity(self, now: float) -> None:
        mid = self.feed.mid
        st = self.portfolio.state
        if mid is None or st is None:
            return
        equity = self.portfolio.equity(mid)
        self.store.record_equity(now, mid, st.base, st.quote, equity)
        self.store.set("last_info", {k: str(v) for k, v in self.last_info.items()})
        log.info(
            "equity=%s base=%s quote=%s mid=%s realized=%s open_orders=%d",
            equity.quantize(Decimal(1)), st.base, st.quote.quantize(Decimal(1)), mid,
            st.realized_pnl.quantize(Decimal(1)), len(self.oms.open_orders),
        )

    def _heartbeat(self) -> None:
        with contextlib.suppress(OSError):
            self.settings.heartbeat_path.write_text(str(int(time.time())))

    async def _shutdown(self) -> None:
        if self.settings.engine.cancel_on_shutdown and self.oms.open_orders:
            log.info("cancelling %d open order(s) on shutdown", len(self.oms.open_orders))
            try:
                await asyncio.wait_for(self.oms.cancel_all(), timeout=30)
            except Exception:
                log.exception("failed to cancel orders on shutdown")
                self.notifier.send("⚠️ لغو سفارش‌ها هنگام توقف ناموفق بود — دستی بررسی کنید")
        self.store.log_event("lifecycle", "ربات متوقف شد")
        st = self.portfolio.state
        mid = self.feed.mid
        if st is not None and mid is not None:
            self._snapshot_equity(time.time())
            self.notifier.send(
                f"🔌 ربات متوقف شد. ارزش کل: {fmt_rial(self.portfolio.equity(mid))} — "
                f"سود تحقق‌یافته: {fmt_rial(st.realized_pnl)}"
            )
        else:
            self.notifier.send("🔌 ربات متوقف شد.")
