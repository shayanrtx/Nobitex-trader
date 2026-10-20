"""Exception hierarchy for exchange interactions.

The split matters for callers:

* :class:`TransientError` — safe to retry later (network, 5xx, rate limit).
* :class:`OrderRejected` — the exchange refused the request; retrying the same
  request will fail again.
* :class:`AuthError` — credentials are wrong/expired; retrying hammers the
  exchange and can get the IP banned, so the bot must stop.
"""

from __future__ import annotations


class ExchangeError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class TransientError(ExchangeError):
    pass


class RateLimitError(TransientError):
    def __init__(self, message: str, backoff_seconds: float) -> None:
        super().__init__(message, code="TooManyRequests")
        self.backoff_seconds = backoff_seconds


class AuthError(ExchangeError):
    pass


class OrderRejected(ExchangeError):
    pass


class OrderNotFound(ExchangeError):
    pass


class UnknownOrderState(TransientError):
    """A write request's outcome is unknown (e.g. timeout after sending)."""
