"""SQLite persistence: orders, fills, equity curve and a small key/value state.

SQLite in WAL mode is plenty for a single bot process; statements are tiny so
they run synchronously on the event loop.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

from nobitex_trader.domain.models import Fill, Order

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    key TEXT PRIMARY KEY,
    exchange_id INTEGER,
    client_order_id TEXT,
    side TEXT NOT NULL,
    price TEXT NOT NULL,
    amount TEXT NOT NULL,
    matched TEXT NOT NULL,
    status TEXT NOT NULL,
    fee TEXT NOT NULL,
    tag TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_key TEXT NOT NULL,
    exchange_id INTEGER,
    side TEXT NOT NULL,
    price TEXT NOT NULL,
    amount TEXT NOT NULL,
    fee TEXT NOT NULL,
    fee_currency TEXT NOT NULL,
    realized_pnl TEXT,
    tag TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL PRIMARY KEY,
    price TEXT NOT NULL,
    base TEXT NOT NULL,
    quote TEXT NOT NULL,
    equity_quote TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    level TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return {"__dec__": str(o)}
        return super().default(o)


def _decode_hook(obj: dict[str, Any]) -> Any:
    if set(obj) == {"__dec__"}:
        return Decimal(obj["__dec__"])
    return obj


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # ----------------------------------------------------------------- key/value

    def get(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"], object_hook=_decode_hook)

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO kv(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value, cls=_DecimalEncoder), time.time()),
        )

    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM kv WHERE key=?", (key,))

    # -------------------------------------------------------------------- orders

    def upsert_order(self, order: Order) -> None:
        self._conn.execute(
            """INSERT INTO orders(key, exchange_id, client_order_id, side, price, amount, matched,
                                  status, fee, tag, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET exchange_id=excluded.exchange_id,
                 matched=excluded.matched, status=excluded.status, fee=excluded.fee,
                 updated_at=excluded.updated_at""",
            (
                order.key,
                order.id,
                order.client_order_id,
                order.side.value,
                str(order.price),
                str(order.amount),
                str(order.matched),
                order.status.value,
                str(order.fee),
                order.tag,
                order.created_at,
                time.time(),
            ),
        )

    def open_order_rows(self) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM orders WHERE status IN ('New','Active','Inactive') "
                "ORDER BY created_at"
            )
        )

    # --------------------------------------------------------------------- fills

    def record_fill(self, fill: Fill, realized_pnl: Decimal | None) -> None:
        self._conn.execute(
            """INSERT INTO fills(order_key, exchange_id, side, price, amount, fee, fee_currency,
                                 realized_pnl, tag, ts) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                fill.order_key,
                fill.order_id,
                fill.side.value,
                str(fill.price),
                str(fill.amount),
                str(fill.fee),
                fill.fee_currency,
                str(realized_pnl) if realized_pnl is not None else None,
                fill.tag,
                fill.timestamp,
            ),
        )

    def fills_since(self, ts: float) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM fills WHERE ts>=? ORDER BY ts", (ts,)))

    def recent_fills(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM fills ORDER BY ts DESC LIMIT ?", (limit,)))

    # -------------------------------------------------------------------- events

    def log_event(
        self, kind: str, message: str, level: str = "info", data: dict[str, Any] | None = None
    ) -> None:
        """Append to the audit journal (what the bot did and why)."""
        self._conn.execute(
            "INSERT INTO events(ts, level, kind, message, data) VALUES(?,?,?,?,?)",
            (time.time(), level, kind, message,
             json.dumps(data, cls=_DecimalEncoder, ensure_ascii=False) if data else None),
        )

    def events(
        self, limit: int = 200, kind: str | None = None, before_id: int | None = None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM events WHERE 1=1"
        args: list[Any] = []
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        if before_id:
            sql += " AND id<?"
            args.append(before_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return list(self._conn.execute(sql, args))

    def prune(self, keep_days: float = 90) -> None:
        cutoff = time.time() - keep_days * 86400
        self._conn.execute("DELETE FROM events WHERE ts<?", (cutoff,))
        self._conn.execute("DELETE FROM equity WHERE ts<?", (cutoff,))

    # -------------------------------------------------------------------- equity

    def record_equity(
        self, ts: float, price: Decimal, base: Decimal, quote: Decimal, equity: Decimal
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO equity(ts, price, base, quote, equity_quote) VALUES(?,?,?,?,?)",
            (ts, str(price), str(base), str(quote), str(equity)),
        )

    def equity_curve(self, since: float = 0) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM equity WHERE ts>=? ORDER BY ts", (since,)))
