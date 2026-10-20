"""Management dashboard: JSON API + a self-contained single-page UI.

Runs inside the bot process (same event loop), so it shows live state and its
controls act on the running engine. Every endpoint requires the dashboard token
(``Authorization: Bearer <token>``). The UI has no external assets (no CDNs,
which are often unreachable from Iran).
"""

from __future__ import annotations

import hmac
import time
from decimal import Decimal
from importlib import resources
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nobitex_trader.engine import TradingEngine


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def create_app(engine: TradingEngine, token: str) -> FastAPI:
    if not token:
        raise ValueError("dashboard token required")
    app = FastAPI(title="Nobitex Trader", docs_url=None, redoc_url=None, openapi_url=None)
    store = engine.store
    package = resources.files("nobitex_trader.dashboard")
    page = package.joinpath("index.html").read_text("utf-8")
    # Bundled Vazirmatn font (OFL) — served locally, no CDN needed.
    app.mount("/static", StaticFiles(directory=str(package.joinpath("static"))), name="static")

    def auth(authorization: str = Header(default="")) -> None:
        supplied = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        # The page itself holds no data; it asks for the token and calls the API.
        return page

    @app.get("/api/summary", dependencies=[Depends(auth)])
    def summary() -> JSONResponse:
        return JSONResponse(_jsonable(engine.status()))

    @app.get("/api/equity", dependencies=[Depends(auth)])
    def equity(hours: float = Query(24, gt=0, le=24 * 365)) -> JSONResponse:
        rows = store.equity_curve(time.time() - hours * 3600)
        step = max(1, len(rows) // 1000)  # keep payload small
        points = [
            {"ts": r["ts"], "equity": float(r["equity_quote"]), "price": float(r["price"]),
             "base": float(r["base"])}
            for r in rows[::step]
        ]
        return JSONResponse(points)

    @app.get("/api/orders", dependencies=[Depends(auth)])
    def orders() -> JSONResponse:
        out = [
            {"side": o.side.value, "price": o.price, "amount": o.amount, "matched": o.matched,
             "status": o.status.value, "tag": o.tag, "id": o.id, "created_at": o.created_at}
            for o in sorted(engine.oms.open_orders, key=lambda o: o.price, reverse=True)
        ]
        return JSONResponse(_jsonable(out))

    @app.get("/api/fills", dependencies=[Depends(auth)])
    def fills(limit: int = Query(100, ge=1, le=1000)) -> JSONResponse:
        out = [
            {"ts": r["ts"], "side": r["side"], "price": float(r["price"]),
             "amount": float(r["amount"]), "fee": float(r["fee"]),
             "fee_currency": r["fee_currency"],
             "pnl": float(r["realized_pnl"]) if r["realized_pnl"] is not None else None,
             "tag": r["tag"]}
            for r in store.recent_fills(limit)
        ]
        return JSONResponse(out)

    @app.get("/api/events", dependencies=[Depends(auth)])
    def events(
        limit: int = Query(200, ge=1, le=2000), kind: str | None = None
    ) -> JSONResponse:
        out = [
            {"id": r["id"], "ts": r["ts"], "level": r["level"], "kind": r["kind"],
             "message": r["message"]}
            for r in store.events(limit, kind or None)
        ]
        return JSONResponse(out)

    @app.get("/api/stats", dependencies=[Depends(auth)])
    def stats(days: float = Query(30, gt=0, le=365)) -> JSONResponse:
        """Per-day realized PnL, fees and trade counts."""
        rows = store.fills_since(time.time() - days * 86400)
        per_day: dict[str, dict[str, float]] = {}
        for r in rows:
            day = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
            d = per_day.setdefault(day, {"pnl": 0.0, "fees": 0.0, "buys": 0, "sells": 0,
                                         "volume": 0.0})
            if r["realized_pnl"] is not None:
                d["pnl"] += float(r["realized_pnl"])
            fee = float(r["fee"])
            d["fees"] += fee * float(r["price"]) if r["fee_currency"] != "rls" else fee
            d["buys" if r["side"] == "buy" else "sells"] += 1
            d["volume"] += float(r["price"]) * float(r["amount"])
        return JSONResponse([{"day": k, **v} for k, v in sorted(per_day.items())])

    @app.post("/api/control/{action}", dependencies=[Depends(auth)])
    def control(action: str) -> JSONResponse:
        if action == "pause":
            engine.pause()
        elif action == "resume":
            engine.resume()
        elif action == "cancel-all":
            engine.request_cancel_all()
        else:
            raise HTTPException(status_code=404, detail="unknown action")
        return JSONResponse({"ok": True, "action": action})

    return app
