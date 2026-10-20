"""Composition root: builds and runs the bot from :class:`Settings`."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import Any

from nobitex_trader.config import Settings
from nobitex_trader.data.market_data import MarketDataFeed
from nobitex_trader.engine import TradingEngine
from nobitex_trader.exchange.auth import ApiKeyAuth, Authenticator, TokenAuth
from nobitex_trader.exchange.base import TradingVenue
from nobitex_trader.exchange.paper import PaperVenue
from nobitex_trader.exchange.rest import NobitexRestClient
from nobitex_trader.exchange.websocket import CentrifugoClient
from nobitex_trader.notify import BotApiNotifier, Notifier
from nobitex_trader.persistence.store import Store
from nobitex_trader.strategy.registry import create_strategy

log = logging.getLogger(__name__)


def build_auth(settings: Settings) -> Authenticator | None:
    if settings.api_key and settings.api_private_key:
        return ApiKeyAuth(settings.api_key, settings.api_private_key.get_secret_value())
    if settings.api_token:
        return TokenAuth(settings.api_token.get_secret_value())
    return None


def build_rest(settings: Settings) -> NobitexRestClient:
    return NobitexRestClient(
        settings.exchange.base_url,
        auth=build_auth(settings),
        user_agent=settings.exchange.user_agent,
        timeout=settings.exchange.timeout_seconds,
    )


def build_notifier(settings: Settings) -> Notifier:
    n = settings.notify
    if n.enabled and settings.notify_bot_token and n.chat_id:
        return BotApiNotifier(
            settings.notify_bot_token.get_secret_value(),
            n.chat_id,
            api_base=n.api_base,
            proxy=n.proxy,
            prefix="[PAPER] " if settings.mode == "paper" else "",
        )
    return Notifier()


def _start_dashboard(
    settings: Settings, engine: TradingEngine
) -> tuple[Any, asyncio.Task[None]] | None:
    if not settings.dashboard.enabled:
        return None
    if not settings.dashboard_token:
        raise SystemExit("dashboard enabled but NBX_DASHBOARD_TOKEN is not set")
    import uvicorn

    from nobitex_trader.dashboard.server import create_app

    app = create_app(engine, settings.dashboard_token.get_secret_value())
    config = uvicorn.Config(
        app,
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve(), name="dashboard")
    log.info("dashboard on http://%s:%d", settings.dashboard.host, settings.dashboard.port)
    return server, task


async def run_bot(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    market = settings.market.spec()
    store = Store(settings.db_path)
    rest = build_rest(settings)
    notifier = build_notifier(settings)
    strategy = create_strategy(settings)
    live = settings.mode == "live"

    ws: CentrifugoClient | None = None
    if settings.exchange.use_websocket:
        token_provider = rest.websocket_token if live else None
        ws = CentrifugoClient(
            settings.exchange.ws_url,
            token_provider=token_provider,
            user_agent=settings.exchange.user_agent,
        )

    feed = MarketDataFeed(
        market.symbol,
        rest,
        ws,
        candle_resolution=settings.grid.candle_resolution,
        candle_history=settings.grid.candle_history,
        ws_stale_after=settings.risk.stale_data_seconds / 2,
    )

    venue: TradingVenue
    if live:
        venue = rest
    else:
        paper = PaperVenue(
            market,
            {market.src: settings.paper.initial_base, market.dst: settings.paper.initial_quote},
            store=store,
        )
        feed.on_book(paper.on_book)
        feed.on_trade(paper.on_trade)
        venue = paper

    engine = TradingEngine(settings, market, feed, venue, strategy, store, notifier)

    if live and ws is not None:
        profile = await rest.profile()
        param = profile.get("websocketAuthParam")
        if param:
            ws.subscribe(f"private:orders#{param}", lambda _c, _d: engine.notify_order_event())
        else:
            log.warning("websocketAuthParam missing; relying on REST polling for order updates")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, engine.stop)

    ws_task = asyncio.create_task(ws.run(), name="ws") if ws else None
    dashboard = _start_dashboard(settings, engine)
    await notifier.start()
    try:
        await feed.start()
        await engine.run()
    finally:
        if dashboard is not None:
            server, task = dashboard
            server.should_exit = True
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=5)
        if ws is not None:
            ws.stop()
        if ws_task is not None:
            ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ws_task
        await feed.stop()
        await notifier.stop()
        await rest.close()
        store.close()
