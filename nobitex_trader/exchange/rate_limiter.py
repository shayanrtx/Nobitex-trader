"""Client-side rate limiting.

Nobitex enforces per-endpoint limits and bans a token for 2 minutes if limits are
ignored repeatedly, so we throttle *before* sending instead of reacting to 429s.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable


class SlidingWindowLimiter:
    """At most ``limit`` acquisitions in any ``window`` seconds."""

    def __init__(
        self, limit: int, window: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if limit <= 0 or window <= 0:
            raise ValueError("limit and window must be positive")
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._blocked_until = 0.0

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window:
            self._events.popleft()

    def available(self) -> int:
        now = self._clock()
        self._prune(now)
        if now < self._blocked_until:
            return 0
        return self.limit - len(self._events)

    def wait_time(self) -> float:
        now = self._clock()
        self._prune(now)
        blocked = max(0.0, self._blocked_until - now)
        if len(self._events) < self.limit:
            return blocked
        return max(blocked, self.window - (now - self._events[0]))

    def try_acquire(self) -> bool:
        if self.wait_time() > 0:
            return False
        self._events.append(self._clock())
        return True

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                delay = self.wait_time()
                if delay <= 0:
                    self._events.append(self._clock())
                    return
                await asyncio.sleep(delay)

    def penalize(self, seconds: float) -> None:
        """Honour a server-provided back-off."""
        self._blocked_until = max(self._blocked_until, self._clock() + seconds)


class RateLimiterRegistry:
    """Named limiters, one per endpoint group."""

    def __init__(self) -> None:
        self._limiters: dict[str, SlidingWindowLimiter] = {}

    def register(self, name: str, limit: int, window: float) -> SlidingWindowLimiter:
        limiter = SlidingWindowLimiter(limit, window)
        self._limiters[name] = limiter
        return limiter

    def get(self, name: str) -> SlidingWindowLimiter | None:
        return self._limiters.get(name)
