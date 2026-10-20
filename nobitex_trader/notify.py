"""Non-blocking notifications through a Telegram-compatible Bot API.

Works with Telegram (``https://api.telegram.org``) and Bale (``https://tapi.bale.ai``,
reachable from inside Iran without a VPN). Messages are queued and sent by a
background task so a slow/blocked messenger can never stall trading.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx

log = logging.getLogger(__name__)


class Notifier:
    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def send(self, text: str) -> None:
        log.info("notify: %s", text)


class BotApiNotifier(Notifier):
    def __init__(
        self,
        token: str,
        chat_id: str,
        api_base: str = "https://api.telegram.org",
        proxy: str | None = None,
        prefix: str = "",
        max_queue: int = 200,
    ) -> None:
        self._url = f"{api_base.rstrip('/')}/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._prefix = prefix
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue)
        self._client = httpx.AsyncClient(timeout=15, proxy=proxy)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker(), name="notifier")

    async def stop(self) -> None:
        # Give queued messages (e.g. "stopped") a moment to go out.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=5)
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await self._client.aclose()

    def send(self, text: str) -> None:
        log.info("notify: %s", text)
        try:
            self._queue.put_nowait(f"{self._prefix}{text}")
        except asyncio.QueueFull:
            log.warning("notification queue full; dropping message")

    async def _worker(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                for attempt in range(3):
                    try:
                        resp = await self._client.post(
                            self._url, json={"chat_id": self._chat_id, "text": text[:4000]}
                        )
                        if resp.status_code == 429:
                            await asyncio.sleep(float(resp.json().get("parameters", {})
                                                      .get("retry_after", 5)))
                            continue
                        resp.raise_for_status()
                        break
                    except httpx.HTTPError as exc:
                        log.warning("notification failed (attempt %d): %r", attempt + 1, exc)
                        await asyncio.sleep(2 * (attempt + 1))
            finally:
                self._queue.task_done()
