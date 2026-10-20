from __future__ import annotations

import base64
import json
from decimal import Decimal

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nobitex_trader.domain.models import OrderRequest, OrderStatus, Side
from nobitex_trader.exchange.auth import ApiKeyAuth, TokenAuth
from nobitex_trader.exchange.errors import (
    AuthError,
    OrderRejected,
    RateLimitError,
    UnknownOrderState,
)
from nobitex_trader.exchange.rate_limiter import SlidingWindowLimiter
from nobitex_trader.exchange.rest import NobitexRestClient, parse_order
from nobitex_trader.exchange.websocket import extract_publication, parse_frame


def client(handler, auth=None) -> NobitexRestClient:  # type: ignore[no-untyped-def]
    return NobitexRestClient(
        "https://apiv2.nobitex.ir",
        auth=auth or TokenAuth("t0ken"),
        transport=httpx.MockTransport(handler),
        max_retries=2,
    )


async def test_orderbook_parsing() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v3/orderbook/USDTIRT"
        assert req.headers["user-agent"].startswith("TraderBot/")
        return httpx.Response(200, json={
            "status": "ok", "lastUpdate": 1644991756704, "lastTradePrice": "1000000",
            "asks": [["1001000", "5"], ["1000500", "2"]],
            "bids": [["999000", "3"], ["999500", "1"]],
        })

    c = client(handler)
    book = await c.orderbook("USDTIRT")
    assert book.best_ask == Decimal("1000500") and book.best_bid == Decimal("999500")
    await c.close()


async def test_place_order_payload_and_auth(market) -> None:  # type: ignore[no-untyped-def]
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"status": "ok", "order": {
            "id": 25, "type": "buy", "price": "1000000", "amount": "10", "matchedAmount": "0",
            "status": "Active", "clientOrderId": "nbxtb1", "fee": "0"}})

    c = client(handler)
    order = await c.place_order(market, OrderRequest(Side.BUY, Decimal("1000000"),
                                                     Decimal("10"), "nbxtb1", "b1"))
    assert seen["auth"] == "Token t0ken"
    assert seen["body"] == {"type": "buy", "execution": "limit", "srcCurrency": "usdt",
                            "dstCurrency": "rls", "amount": "10", "price": "1000000",
                            "clientOrderId": "nbxtb1"}
    assert order.id == 25 and order.status is OrderStatus.ACTIVE and order.tag == "b1"
    await c.close()


async def test_order_rejection_maps_to_exception(market) -> None:  # type: ignore[no-untyped-def]
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "failed", "code": "SmallOrder",
                                         "message": "too small"})

    c = client(handler)
    with pytest.raises(OrderRejected) as exc:
        await c.place_order(market, OrderRequest(Side.BUY, Decimal(1), Decimal(1), "x"))
    assert exc.value.code == "SmallOrder"
    await c.close()


async def test_write_timeout_is_unknown_state_not_retried(market) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=req)

    c = client(handler)
    with pytest.raises(UnknownOrderState):
        await c.place_order(market, OrderRequest(Side.BUY, Decimal(1), Decimal(1), "x"))
    assert calls == 1  # never blindly re-sent
    await c.close()


async def test_duplicate_client_id_adopts_existing(market) -> None:  # type: ignore[no-untyped-def]
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/add"):
            return httpx.Response(200, json={"status": "failed", "code": "DuplicateClientOrderId",
                                             "message": "dup"})
        return httpx.Response(200, json={"status": "ok", "order": {
            "id": 7, "type": "sell", "price": "5", "amount": "1", "matchedAmount": "0",
            "status": "Active", "clientOrderId": "abc"}})

    c = client(handler)
    order = await c.place_order(market, OrderRequest(Side.SELL, Decimal(5), Decimal(1), "abc"))
    assert order.id == 7
    await c.close()


async def test_rate_limit_and_auth_errors() -> None:
    responses = iter([
        httpx.Response(429, json={"status": "failed", "code": "TooManyRequests", "backOff": 0}),
        httpx.Response(401, json={"detail": "bad token"}),
    ])

    c = client(lambda req: next(responses))
    with pytest.raises(AuthError):  # 429 retried once (backOff=0), then 401
        await c.orderbook("USDTIRT")
    await c.close()

    c = client(lambda req: httpx.Response(429, json={"code": "TooManyRequests", "backOff": 0}))
    with pytest.raises(RateLimitError):
        await c.orderbook("USDTIRT")
    await c.close()


def test_api_key_signature_verifies() -> None:
    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes_raw()
    auth = ApiKeyAuth("PUB", base64.urlsafe_b64encode(seed).decode(), clock=lambda: 1700000000)
    headers = auth.headers("POST", "/market/orders/add", '{"a":1}')
    assert headers["Nobitex-Key"] == "PUB"
    assert headers["Nobitex-Timestamp"] == "1700000000"
    key.public_key().verify(
        base64.b64decode(headers["Nobitex-Signature"]),
        b'1700000000POST/market/orders/add{"a":1}',
    )


def test_parse_order_variants() -> None:
    o = parse_order({"id": 1, "type": "sell", "price": "market", "amount": "3.0000000000",
                     "unmatchedAmount": "1", "status": "Active", "fee": "0E-10"})
    assert o.matched == Decimal(2) and o.price == 0 and o.side is Side.SELL
    ws = parse_order({"orderId": 9, "side": "Buy", "amount": "2", "filledAmount": "2",
                      "status": "Done", "avgFilledPrice": "1000"})
    assert ws.id == 9 and ws.matched == 2 and ws.avg_price == 1000


def test_centrifugo_frame_parsing() -> None:
    frame = ('{}\n{"push":{"channel":"public:orderbook-USDTIRT","pub":{"data":'
             '"{\\"asks\\": [], \\"bids\\": []}","offset":1}}}\n')
    msgs = parse_frame(frame)
    assert msgs[0] == {}
    assert extract_publication(msgs[1]) == ("public:orderbook-USDTIRT", {"asks": [], "bids": []})
    assert extract_publication({"id": 2, "subscribe": {}}) is None


def test_sliding_window_limiter() -> None:
    now = [0.0]
    lim = SlidingWindowLimiter(2, 10, clock=lambda: now[0])
    assert lim.try_acquire() and lim.try_acquire()
    assert not lim.try_acquire()
    assert lim.wait_time() == pytest.approx(10)
    now[0] = 10.01
    assert lim.try_acquire()
    lim.penalize(5)
    assert lim.available() == 0


def test_account_fee_from_profile(market, store) -> None:  # type: ignore[no-untyped-def]
    from nobitex_trader.app import apply_account_fees
    from nobitex_trader.exchange.rest import account_fee
    from nobitex_trader.notify import Notifier

    profile = {"profile": {"options": {"fee": "0.35", "feeUsdt": "0.2"}}}
    assert account_fee(profile, "rls") == Decimal("0.0035")
    assert account_fee(profile, "usdt") == Decimal("0.002")
    assert account_fee({}, "rls") is None
    updated = apply_account_fees(market, profile, store, Notifier())
    assert updated.maker_fee == updated.taker_fee == Decimal("0.0035")
    assert any(r["kind"] == "fees" for r in store.events())


def test_spacing_floor_follows_detected_fee(market) -> None:  # type: ignore[no-untyped-def]
    import time
    from dataclasses import replace

    from nobitex_trader.config import GridSettings
    from nobitex_trader.strategy.adaptive_grid import AdaptiveGridStrategy
    from nobitex_trader.strategy.base import InventorySnapshot, StrategyContext
    from tests.conftest import flat_candles, make_book

    pricey = replace(market, maker_fee=Decimal("0.004"), taker_fee=Decimal("0.004"))
    d = AdaptiveGridStrategy(GridSettings()).decide(StrategyContext(
        pricey, make_book("999000", "1001000"), flat_candles(150),
        InventorySnapshot(Decimal(250), Decimal(250_000_000), None), time.time()))
    assert d.info["spacing_pct"] >= Decimal("1.0")  # 2 × 0.4% × 1.25
