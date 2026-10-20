"""Request authentication strategies.

Nobitex supports two schemes:

* **Token** — ``Authorization: Token <hex>``, obtained from the account settings page.
* **API key** (experimental on Nobitex' side) — scoped permissions (READ/TRADE/WITHDRAW)
  and IP whitelist. Each request carries ``Nobitex-Key``, ``Nobitex-Timestamp`` and an
  Ed25519 ``Nobitex-Signature`` over ``timestamp + method + path_with_query + body``.

For a bot the API key is preferable: create it with READ+TRADE only (never WITHDRAW)
and whitelist your IP.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class Authenticator(ABC):
    @abstractmethod
    def headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        """Headers to attach to a private request."""


class TokenAuth(Authenticator):
    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("empty API token")
        self._token = token

    def headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        return {"Authorization": f"Token {self._token}"}


class ApiKeyAuth(Authenticator):
    def __init__(
        self,
        public_key: str,
        private_key: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._public_key = public_key
        self._signer = Ed25519PrivateKey.from_private_bytes(_decode_key(private_key))
        self._clock = clock

    def sign(self, timestamp: str, method: str, path_with_query: str, body: str) -> str:
        message = f"{timestamp}{method.upper()}{path_with_query}{body}".encode()
        return base64.b64encode(self._signer.sign(message)).decode()

    def headers(self, method: str, path_with_query: str, body: str) -> dict[str, str]:
        timestamp = str(int(self._clock()))
        return {
            "Nobitex-Key": self._public_key,
            "Nobitex-Timestamp": timestamp,
            "Nobitex-Signature": self.sign(timestamp, method, path_with_query, body),
        }


def _decode_key(raw: str) -> bytes:
    """Accept url-safe or standard base64, or hex, for a 32-byte Ed25519 seed."""
    raw = raw.strip()
    candidates: list[bytes] = []
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            candidates.append(decoder(raw + "=" * (-len(raw) % 4)))
        except (binascii.Error, ValueError):
            continue
    with contextlib.suppress(ValueError):
        candidates.append(bytes.fromhex(raw))
    for key in candidates:
        if len(key) == 32:
            return key
    raise ValueError("private key must decode to a 32-byte Ed25519 seed")
