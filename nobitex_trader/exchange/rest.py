"""Async REST client for the Nobitex public and private API.

Reference: https://apidocs.nobitex.ir (source: github.com/nobitex/docs-api).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode

import httpx

from nobitex_trader.domain.market import MarketSpec
from nobitex_trader.domain.models import (
    ZERO,
    Balance,
    Candle,
    Order,
    OrderBook,
    OrderRequest,
    OrderStatus,
    Side,
)
from nobitex_trader.exchange.auth import Authenticator
from nobitex_trader.exchange.errors import (
    AuthError,
    ExchangeError,
    OrderNotFound,
    OrderRejected,
    RateLimitError,
    TransientError,
    UnknownOrderState,
)
from nobitex_trader.exchange.rate_limiter import RateLimiterRegistry

log = logging.getLogger(__name__)

# (limit, window seconds) per documented endpoint limit, with ~10% headroom.
_LIMITS: dict[str, tuple[int, float]] = {
    "orderbook": (270, 60),
    "trades": (54, 60),
    "candles": (60, 60),
    "order_add": (270, 600),  # shared across all order-placing APIs
    "order_status": (270, 60),
    "order_list": (27, 60),
    "order_cancel": (80, 60),
    "cancel_old": (27, 60),
    "wallets": (13, 60),
    "profile": (10, 60),
    "ws_token": (10, 60),
}

# Order API error codes that mean "this request will never succeed as-is".
_REJECT_CODES = {
    "InvalidOrderPrice",
    "BadPrice",
    "PriceConditionFailed",
    "OverValueOrder",
    "SmallOrder",
    "DuplicateOrder",
    "InvalidMarketPair",
    "MarketClosed",
    "TradingUnavailable",
    "FeatureUnavailable",
    "DuplicateClientOrderId",
    "NullIdAndClientOrderId",
}


def to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    if value is None or value == "" or value == "market":
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default


def _parse_ts(raw: Any) -> float:
    if isinstance(raw, int | float):
        return float(raw) / (1000.0 if raw > 1e11 else 1.0)
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def parse_order(data: dict[str, Any]) -> Order:
    side_raw = str(data.get("type") or data.get("side") or "buy").lower()
    amount = to_decimal(data.get("amount"))
    if "matchedAmount" in data:
        matched = to_decimal(data.get("matchedAmount"))
    elif "filledAmount" in data:
        matched = to_decimal(data.get("filledAmount"))
    else:
        matched = amount - to_decimal(data.get("unmatchedAmount"), amount)
    avg = to_decimal(data.get("averagePrice") or data.get("avgFilledPrice"))
    order_id = data.get("id", data.get("orderId"))
    return Order(
        id=int(order_id) if order_id is not None else None,
        client_order_id=data.get("clientOrderId") or None,
        side=Side.SELL if side_raw == "sell" else Side.BUY,
        price=to_decimal(data.get("price")),
        amount=amount,
        matched=matched,
        status=OrderStatus.parse(data.get("status")),
        fee=to_decimal(data.get("fee")),
        avg_price=avg if avg > ZERO else None,
        created_at=_parse_ts(data.get("created_at")),
    )


def account_fee(profile: dict[str, Any], dst: str) -> Decimal | None:
    """Trading fee (fraction) for the user's tier from ``/users/profile``.

    Nobitex reports percentages: ``options.fee`` for rial markets and
    ``options.feeUsdt`` for tether markets (e.g. ``"0.35"`` → 0.0035).
    """
    options = (profile.get("profile") or {}).get("options") or {}
    raw = options.get("fee") if dst.lower() in ("rls", "irt") else options.get("feeUsdt")
    pct = to_decimal(raw, Decimal(-1))
    if pct < 0 or pct >= 5:
        return None
    return pct / 100


def parse_orderbook(data: dict[str, Any], received_ts: float | None = None) -> OrderBook:
    def levels(raw: Any) -> tuple[tuple[Decimal, Decimal], ...]:
        out = []
        for entry in raw or ():
            price, amount = to_decimal(entry[0]), to_decimal(entry[1])
            if price > ZERO and amount > ZERO:
                out.append((price, amount))
        return tuple(out)

    bids = tuple(sorted(levels(data.get("bids")), key=lambda x: x[0], reverse=True))
    asks = tuple(sorted(levels(data.get("asks")), key=lambda x: x[0]))
    last = to_decimal(data.get("lastTradePrice"))
    now = time.time()
    return OrderBook(
        bids=bids,
        asks=asks,
        last_trade_price=last if last > ZERO else None,
        exchange_ts=_parse_ts(data.get("lastUpdate")) if data.get("lastUpdate") else now,
        received_ts=received_ts if received_ts is not None else now,
    )


class NobitexRestClient:
    """Implements both :class:`MarketDataSource` and :class:`TradingVenue`."""

    def __init__(
        self,
        base_url: str,
        auth: Authenticator | None = None,
        user_agent: str = "TraderBot/NobitexTrader",
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        self._auth = auth
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            transport=transport,
        )
        self.limits = RateLimiterRegistry()
        for name, (limit, window) in _LIMITS.items():
            self.limits.register(name, limit, window)

    @property
    def has_auth(self) -> bool:
        return self._auth is not None

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ transport

    async def _request(
        self,
        method: str,
        path: str,
        *,
        limiter: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        private: bool = False,
        idempotent: bool = True,
    ) -> dict[str, Any]:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None})
        path_with_query = f"{path}?{query}" if query else path
        body = json.dumps(payload, separators=(",", ":")) if payload is not None else ""
        attempts = self._max_retries if idempotent else 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            rl = self.limits.get(limiter)
            if rl is not None:
                await rl.acquire()
            headers: dict[str, str] = {}
            if body:
                headers["Content-Type"] = "application/json"
            if private:
                if self._auth is None:
                    raise AuthError("private endpoint called without credentials")
                headers.update(self._auth.headers(method, path_with_query, body))
            try:
                resp = await self._client.request(
                    method, path_with_query, content=body or None, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if not idempotent:
                    # The request may have reached the exchange: outcome unknown.
                    raise UnknownOrderState(f"{method} {path}: {exc!r}") from exc
                await asyncio.sleep(min(2**attempt, 10))
                continue

            try:
                return self._handle_response(resp, limiter)
            except RateLimitError as exc:
                last_error = exc
                if not idempotent or attempt == attempts:
                    raise
                await asyncio.sleep(exc.backoff_seconds)
            except TransientError as exc:
                last_error = exc
                if not idempotent:
                    # A 5xx after sending a write: it may or may not have been applied.
                    raise UnknownOrderState(f"{method} {path}: {exc}") from exc
                if attempt == attempts:
                    raise
                await asyncio.sleep(min(2**attempt, 10))

        raise TransientError(f"{method} {path} failed after retries: {last_error!r}")

    def _handle_response(self, resp: httpx.Response, limiter: str) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {"data": data}

        code = data.get("code")
        message = str(data.get("message") or data.get("detail") or resp.reason_phrase)

        if resp.status_code == 429 or code == "TooManyRequests":
            backoff = float(data.get("backOff") or 10)
            rl = self.limits.get(limiter)
            if rl is not None:
                rl.penalize(backoff)
            raise RateLimitError(message, backoff)
        if resp.status_code in (401, 403):
            raise AuthError(f"HTTP {resp.status_code}: {message}", code)
        if resp.status_code == 404:
            raise OrderNotFound(message, code)
        if resp.status_code >= 500:
            raise TransientError(f"HTTP {resp.status_code}: {message}", code)
        if resp.status_code >= 400:
            raise ExchangeError(f"HTTP {resp.status_code}: {message}", code)

        status = data.get("status") or data.get("s")
        if status in ("failed", "error"):
            if code in _REJECT_CODES:
                raise OrderRejected(f"{code}: {message}", code)
            if code in ("NotFound", "OrderNotFound"):
                raise OrderNotFound(message, code)
            raise ExchangeError(f"{code}: {message}", code)
        return data

    # --------------------------------------------------------------- market data

    async def orderbook(self, symbol: str) -> OrderBook:
        data = await self._request("GET", f"/v3/orderbook/{symbol}", limiter="orderbook")
        return parse_orderbook(data)

    async def candles(
        self, symbol: str, resolution: str, countback: int, to_ts: int | None = None
    ) -> list[Candle]:
        data = await self._request(
            "GET",
            "/market/udf/history",
            limiter="candles",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "to": to_ts or int(time.time()),
                "countback": countback,
            },
        )
        if data.get("s") == "no_data":
            return []
        keys = ("t", "o", "h", "l", "c", "v")
        series = [data.get(k) or [] for k in keys]
        return [
            Candle(
                ts=int(t),
                open=to_decimal(o),
                high=to_decimal(h),
                low=to_decimal(lo),
                close=to_decimal(c),
                volume=to_decimal(v),
            )
            for t, o, h, lo, c, v in zip(*series, strict=False)
        ]

    async def candles_range(
        self, symbol: str, resolution: str, from_ts: int, to_ts: int
    ) -> list[Candle]:
        """All candles in ``[from_ts, to_ts]``, following pagination (500 per page)."""
        out: dict[int, Candle] = {}
        page = 1
        while True:
            data = await self._request(
                "GET",
                "/market/udf/history",
                limiter="candles",
                params={"symbol": symbol, "resolution": resolution, "from": from_ts,
                        "to": to_ts, "page": page},
            )
            if data.get("s") != "ok":
                break
            ts_list = data.get("t") or []
            for t, o, h, lo, c, v in zip(ts_list, data["o"], data["h"], data["l"], data["c"],
                                         data["v"], strict=False):
                out[int(t)] = Candle(int(t), to_decimal(o), to_decimal(h), to_decimal(lo),
                                     to_decimal(c), to_decimal(v))
            if len(ts_list) < 500:
                break
            page += 1
        return [out[k] for k in sorted(out)]

    # ------------------------------------------------------------------- trading

    async def place_order(self, market: MarketSpec, request: OrderRequest) -> Order:
        payload = {
            "type": request.side.value,
            "execution": "limit",
            "srcCurrency": market.src,
            "dstCurrency": market.dst,
            "amount": str(request.amount),
            "price": str(request.price),
            "clientOrderId": request.client_order_id,
        }
        try:
            data = await self._request(
                "POST",
                "/market/orders/add",
                limiter="order_add",
                payload=payload,
                private=True,
                idempotent=False,
            )
        except OrderRejected as exc:
            if exc.code == "DuplicateClientOrderId":
                # A previous attempt actually went through — adopt it.
                return await self.get_order(market, client_order_id=request.client_order_id)
            raise
        order = parse_order(data["order"])
        return order.with_updates(tag=request.tag)

    async def get_order(
        self, market: MarketSpec, order_id: int | None = None, client_order_id: str | None = None
    ) -> Order:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["id"] = order_id
        if client_order_id:
            payload["clientOrderId"] = client_order_id
        if not payload:
            raise ValueError("order_id or client_order_id required")
        data = await self._request(
            "POST", "/market/orders/status", limiter="order_status", payload=payload, private=True
        )
        if not data.get("order"):
            raise OrderNotFound(f"order {order_id or client_order_id} not found")
        return parse_order(data["order"])

    async def cancel_order(self, market: MarketSpec, order: Order) -> Order | None:
        payload: dict[str, Any] = {"status": "canceled"}
        if order.id is not None:
            payload["order"] = order.id
        elif order.client_order_id:
            payload["clientOrderId"] = order.client_order_id
        else:
            raise ValueError("cannot cancel an order without id")
        # Cancelling twice is harmless, so retries are allowed.
        data = await self._request(
            "POST",
            "/market/orders/update-status",
            limiter="order_cancel",
            payload=payload,
            private=True,
        )
        raw = data.get("order")
        return parse_order(raw).with_updates(tag=order.tag) if raw else None

    async def open_orders(self, market: MarketSpec) -> list[Order]:
        data = await self._request(
            "GET",
            "/market/orders/list",
            limiter="order_list",
            params={
                "status": "open",
                "srcCurrency": market.src,
                "dstCurrency": market.dst,
                "details": 2,
                "tradeType": "spot",
            },
            private=True,
        )
        return [parse_order(o) for o in data.get("orders") or []]

    async def cancel_all(self, market: MarketSpec) -> None:
        await self._request(
            "POST",
            "/market/orders/cancel-old",
            limiter="cancel_old",
            payload={"srcCurrency": market.src, "dstCurrency": market.dst, "tradeType": "spot"},
            private=True,
        )

    async def balances(self, currencies: list[str]) -> dict[str, Balance]:
        data = await self._request(
            "GET",
            "/v2/wallets",
            limiter="wallets",
            params={"currencies": ",".join(c.lower() for c in currencies)},
            private=True,
        )
        out: dict[str, Balance] = {}
        for code, wallet in (data.get("wallets") or {}).items():
            out[code.lower()] = Balance(
                currency=code.lower(),
                total=to_decimal(wallet.get("balance")),
                blocked=to_decimal(wallet.get("blocked")),
            )
        for c in currencies:
            out.setdefault(c.lower(), Balance(currency=c.lower(), total=ZERO))
        return out

    # ----------------------------------------------------------------- websocket

    async def profile(self) -> dict[str, Any]:
        return await self._request("GET", "/users/profile", limiter="profile", private=True)

    async def websocket_token(self) -> str:
        data = await self._request("GET", "/auth/ws/token/", limiter="ws_token", private=True)
        return str(data["token"])
