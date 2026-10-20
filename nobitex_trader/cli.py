"""Command-line interface.

    nobitex-trader run                 # start the bot (paper or live, from config)
    nobitex-trader check               # connectivity, credentials, websocket, dry-run quotes
    nobitex-trader backtest --days 30  # fetch history and backtest the configured strategy
    nobitex-trader fetch-candles --days 60 --out data/usdtirt_1m.csv
    nobitex-trader status              # portfolio, open orders, recent fills, risk state
    nobitex-trader resume              # clear a risk halt
    nobitex-trader cancel-all          # cancel every bot order on the exchange
    nobitex-trader healthcheck         # exit 0 if the bot loop is alive (Docker)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from nobitex_trader.config import Settings, load_settings
from nobitex_trader.logging_setup import setup_logging


def _settings() -> Settings:
    try:
        return load_settings()
    except ValidationError as exc:
        print(f"Invalid configuration:\n{exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def cmd_run(settings: Settings, _args: argparse.Namespace) -> int:
    from nobitex_trader.app import run_bot

    if settings.mode == "live":
        print("⚠️  LIVE MODE — real orders will be placed.")
    asyncio.run(run_bot(settings))
    return 0


async def _check(settings: Settings) -> int:
    from nobitex_trader.app import build_rest
    from nobitex_trader.exchange.errors import ExchangeError
    from nobitex_trader.exchange.websocket import CentrifugoClient
    from nobitex_trader.strategy.base import InventorySnapshot, StrategyContext
    from nobitex_trader.strategy.registry import create_strategy

    market = settings.market.spec()
    rest = build_rest(settings)
    ok = True
    try:
        print(f"mode={settings.mode} market={market.symbol} strategy={settings.strategy}")
        book = await rest.orderbook(market.symbol)
        print(f"✅ REST market data: bid={book.best_bid} ask={book.best_ask} "
              f"spread={(book.spread_ratio or 0) * 100:.3f}%")
        candles = await rest.candles(market.symbol, settings.grid.candle_resolution,
                                     settings.grid.candle_history)
        print(f"✅ candles: {len(candles)} x {settings.grid.candle_resolution}")

        if settings.api_token or settings.api_key:
            try:
                bal = await rest.balances([market.src, market.dst])
                print(f"✅ auth OK — {market.src}: {bal[market.src].free} free, "
                      f"{market.dst}: {bal[market.dst].free} free")
            except ExchangeError as exc:
                ok = False
                print(f"❌ auth/balances failed: {exc}")
        else:
            print("ℹ️  no credentials configured (fine for paper mode)")

        ws = CentrifugoClient(settings.exchange.ws_url, user_agent=settings.exchange.user_agent)
        got = asyncio.Event()
        ws.subscribe(f"public:orderbook-{market.symbol}", lambda _c, _d: got.set())
        task = asyncio.create_task(ws.run())
        try:
            await asyncio.wait_for(got.wait(), timeout=20)
            print("✅ websocket: receiving order book updates")
        except TimeoutError:
            print("⚠️  websocket: no update within 20s (bot will fall back to REST polling)")
        finally:
            ws.stop()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        strategy = create_strategy(settings)
        mid = book.mid or Decimal(0)
        inv = InventorySnapshot(settings.paper.initial_base, settings.paper.initial_quote, mid)
        if settings.mode == "live" and (settings.api_token or settings.api_key):
            inv = InventorySnapshot(
                min(settings.capital.max_base, bal[market.src].free),
                min(settings.capital.max_quote, bal[market.dst].free), mid)
        decision = strategy.decide(StrategyContext(market, book, candles, inv, time.time()))
        print("strategy info:", json.dumps({k: str(v) for k, v in decision.info.items()},
                                           ensure_ascii=False))
        for q in sorted(decision.quotes, key=lambda q: q.price, reverse=True):
            print(f"   {q.side.value:<4} {q.amount:>10} @ {q.price:>12}  ({q.tag})")
    except Exception as exc:
        ok = False
        print(f"❌ check failed: {exc!r}")
    finally:
        await rest.close()
    return 0 if ok else 1


def cmd_check(settings: Settings, _args: argparse.Namespace) -> int:
    return asyncio.run(_check(settings))


async def _fetch(settings: Settings, days: float) -> list:  # type: ignore[type-arg]
    from nobitex_trader.app import build_rest

    rest = build_rest(settings)
    try:
        now = int(time.time())
        return await rest.candles_range(settings.market.spec().symbol, "1",
                                        now - int(days * 86400), now)
    finally:
        await rest.close()


def cmd_fetch(settings: Settings, args: argparse.Namespace) -> int:
    from nobitex_trader.backtest import save_csv

    candles = asyncio.run(_fetch(settings, args.days))
    save_csv(Path(args.out), candles)
    print(f"saved {len(candles)} candles to {args.out}")
    return 0


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from nobitex_trader.backtest import load_csv, run_backtest, save_csv, synthetic_candles
    from nobitex_trader.strategy.registry import create_strategy

    if args.synthetic:
        candles = synthetic_candles(int(args.days * 1440))
    elif args.csv:
        candles = load_csv(Path(args.csv))
    else:
        candles = asyncio.run(_fetch(settings, args.days))
        if args.save:
            save_csv(Path(args.save), candles)
    print(f"backtesting {len(candles)} one-minute candles…")
    result = asyncio.run(
        run_backtest(settings, create_strategy(settings), candles, Decimal(str(args.spread)))
    )
    print(result.summary())
    return 0


def cmd_status(settings: Settings, _args: argparse.Namespace) -> int:
    from nobitex_trader.persistence.store import Store

    if not settings.db_path.exists():
        print(f"no database at {settings.db_path}")
        return 1
    store = Store(settings.db_path)
    p = store.get("portfolio") or {}
    print("portfolio:", json.dumps({k: str(v) for k, v in p.items()}, ensure_ascii=False))
    print("risk     :", json.dumps({k: str(v) for k, v in (store.get("risk") or {}).items()}))
    print("strategy :", json.dumps(store.get("last_info") or {}, ensure_ascii=False))
    rows = store.open_order_rows()
    print(f"open orders ({len(rows)}):")
    for r in rows:
        print(f"   {r['side']:<4} {r['amount']:>10} @ {r['price']:>12} matched={r['matched']} "
              f"[{r['tag']}]")
    print("recent fills:")
    for r in store.recent_fills(10):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        print(f"   {when} {r['side']:<4} {r['amount']:>10} @ {r['price']:>12} "
              f"pnl={r['realized_pnl']}")
    store.close()
    return 0


def cmd_resume(settings: Settings, _args: argparse.Namespace) -> int:
    from nobitex_trader.persistence.store import Store
    from nobitex_trader.risk import RiskManager

    store = Store(settings.db_path)
    RiskManager(settings.risk, store).resume()
    store.close()
    print("risk halt cleared; restart the bot (or it resumes on its next tick if running)")
    return 0


async def _cancel_all(settings: Settings) -> int:
    from nobitex_trader.app import build_rest

    rest = build_rest(settings)
    market = settings.market.spec()
    prefix = settings.engine.client_order_prefix
    try:
        orders = [o for o in await rest.open_orders(market)
                  if (o.client_order_id or "").startswith(prefix)]
        for order in orders:
            await rest.cancel_order(market, order)
            print(f"cancelled {order.side.value} {order.amount} @ {order.price}")
        print(f"{len(orders)} order(s) cancelled")
    finally:
        await rest.close()
    return 0


def cmd_cancel_all(settings: Settings, _args: argparse.Namespace) -> int:
    return asyncio.run(_cancel_all(settings))


def cmd_healthcheck(settings: Settings, args: argparse.Namespace) -> int:
    try:
        age = time.time() - int(settings.heartbeat_path.read_text().strip())
    except (OSError, ValueError):
        return 1
    return 0 if age <= args.max_age else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nobitex-trader", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("check")
    bt = sub.add_parser("backtest")
    bt.add_argument("--days", type=float, default=14)
    bt.add_argument("--csv", help="1-minute candles CSV (ts,open,high,low,close,volume)")
    bt.add_argument("--save", help="save fetched candles to this CSV")
    bt.add_argument("--spread", type=float, default=0.05, help="assumed half-spread %%")
    bt.add_argument("--synthetic", action="store_true", help="use generated data (demo)")
    fc = sub.add_parser("fetch-candles")
    fc.add_argument("--days", type=float, default=30)
    fc.add_argument("--out", default="data/candles_1m.csv")
    sub.add_parser("status")
    sub.add_parser("resume")
    sub.add_parser("cancel-all")
    hc = sub.add_parser("healthcheck")
    hc.add_argument("--max-age", type=float, default=120)
    args = parser.parse_args(argv)

    settings = _settings()
    if args.command != "healthcheck":
        setup_logging(settings.log_level, settings.log_json)
    handlers = {
        "run": cmd_run,
        "check": cmd_check,
        "backtest": cmd_backtest,
        "fetch-candles": cmd_fetch,
        "status": cmd_status,
        "resume": cmd_resume,
        "cancel-all": cmd_cancel_all,
        "healthcheck": cmd_healthcheck,
    }
    try:
        return handlers[args.command](settings, args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
