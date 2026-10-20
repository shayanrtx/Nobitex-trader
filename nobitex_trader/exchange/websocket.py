"""Minimal Centrifugo (JSON protocol) client for the Nobitex websocket.

Implemented directly on ``websockets`` instead of the SDK so behaviour is explicit
and testable:

* connect → ``{"id": 1, "connect": {"token": ...}}``
* subscribe → ``{"id": n, "subscribe": {"channel": ...}}`` (no fossil delta)
* server ping is an empty ``{}`` message and must be answered with ``{}``
* one frame may carry several newline-separated JSON messages
* automatic reconnect with exponential back-off + jitter, and a watchdog that
  reconnects if the server goes silent (it pings every ~25s)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import websockets

log = logging.getLogger(__name__)

Handler = Callable[[str, dict[str, Any]], Awaitable[None] | None]
TokenProvider = Callable[[], Awaitable[str | None]]


def parse_frame(frame: str | bytes) -> list[dict[str, Any]]:
    """Split a frame into Centrifugo messages (newline-delimited JSON)."""
    if isinstance(frame, bytes):
        frame = frame.decode("utf-8")
    messages: list[dict[str, Any]] = []
    for line in frame.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            log.warning("ws: dropping malformed message: %.200s", line)
            continue
        if isinstance(obj, dict):
            messages.append(obj)
    return messages


def extract_publication(message: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Return ``(channel, data)`` for a push publication, else ``None``."""
    push = message.get("push")
    if not isinstance(push, dict):
        return None
    pub = push.get("pub")
    channel = push.get("channel")
    if not isinstance(pub, dict) or not isinstance(channel, str):
        return None
    data = pub.get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return channel, data


class CentrifugoClient:
    def __init__(
        self,
        url: str,
        token_provider: TokenProvider | None = None,
        user_agent: str = "TraderBot/NobitexTrader",
        silence_timeout: float = 40.0,
        max_backoff: float = 30.0,
    ) -> None:
        self._url = url
        self._token_provider = token_provider
        self._user_agent = user_agent
        self._silence_timeout = silence_timeout
        self._max_backoff = max_backoff
        self._handlers: dict[str, list[Handler]] = {}
        self._next_id = 1
        self._stopped = asyncio.Event()
        self.connected = asyncio.Event()
        self.last_message_at = 0.0
        self.reconnects = 0

    def subscribe(self, channel: str, handler: Handler) -> None:
        self._handlers.setdefault(channel, []).append(handler)

    def stop(self) -> None:
        self._stopped.set()

    def _cmd_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def run(self) -> None:
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("ws: session ended: %r", exc)
            finally:
                self.connected.clear()
            if self._stopped.is_set():
                break
            self.reconnects += 1
            delay = backoff * (0.5 + random.random())
            log.info("ws: reconnecting in %.1fs", delay)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), timeout=delay)
            backoff = min(backoff * 2, self._max_backoff)

    async def _session(self) -> None:
        token = await self._token_provider() if self._token_provider else None
        async with websockets.connect(
            self._url,
            user_agent_header=self._user_agent,
            ping_interval=None,  # Centrifugo uses app-level pings
            open_timeout=15,
            close_timeout=5,
            max_size=8 * 1024 * 1024,
        ) as ws:
            connect: dict[str, Any] = {"name": "nobitex-trader"}
            if token:
                connect["token"] = token
            await ws.send(json.dumps({"id": 1, "connect": connect}))
            reply = await asyncio.wait_for(ws.recv(), timeout=15)
            for msg in parse_frame(reply):
                if "error" in msg:
                    raise ConnectionError(f"connect rejected: {msg['error']}")

            for channel in self._handlers:
                await ws.send(json.dumps({"id": self._cmd_id(), "subscribe": {"channel": channel}}))

            self.connected.set()
            self.last_message_at = time.time()
            log.info("ws: connected, %d channel(s)", len(self._handlers))

            while not self._stopped.is_set():
                try:
                    frame = await asyncio.wait_for(ws.recv(), timeout=self._silence_timeout)
                except TimeoutError as exc:
                    raise ConnectionError("no data from server (watchdog)") from exc
                self.last_message_at = time.time()
                for msg in parse_frame(frame):
                    if not msg:
                        await ws.send("{}")  # pong
                        continue
                    if "error" in msg:
                        log.warning("ws: server error: %s", msg["error"])
                        continue
                    await self._dispatch(msg)

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        pub = extract_publication(msg)
        if pub is None:
            return
        channel, data = pub
        for handler in self._handlers.get(channel, ()):
            try:
                result = handler(channel, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("ws: handler for %s failed", channel)
