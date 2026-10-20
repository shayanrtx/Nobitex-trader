from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from nobitex_trader.config import Settings
from nobitex_trader.domain.models import OrderStatus, Side
from tests.conftest import make_book


def test_price_rounding_never_crosses(market) -> None:  # type: ignore[no-untyped-def]
    assert market.round_price(Decimal("1000005"), Side.BUY) == Decimal("1000000")
    assert market.round_price(Decimal("1000001"), Side.SELL) == Decimal("1000010")
    assert market.round_amount(Decimal("12.3456")) == Decimal("12.34")
    assert str(market.round_price(Decimal("1234567.8"), Side.BUY)) == "1234560"


def test_min_notional(market) -> None:  # type: ignore[no-untyped-def]
    assert not market.meets_min_notional(Decimal("1000000"), Decimal("2.99"))
    assert market.meets_min_notional(Decimal("1000000"), Decimal("3"))


def test_order_status_parsing() -> None:
    assert OrderStatus.parse("Active") is OrderStatus.ACTIVE
    assert OrderStatus.parse("done") is OrderStatus.DONE
    assert OrderStatus.parse("canceled").is_open is False
    assert OrderStatus.parse("SomethingNew") is OrderStatus.ACTIVE  # keep tracking unknowns


def test_orderbook_metrics() -> None:
    book = make_book("1000000", "1002000")
    assert book.mid == Decimal("1001000")
    assert book.spread_ratio == Decimal("0.002")


def test_live_mode_requires_credentials(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError):
        Settings(mode="live", data_dir=tmp_path)
    Settings(mode="live", api_token="abc", data_dir=tmp_path)


def test_spacing_must_exceed_fees(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValidationError, match="round-trip"):
        Settings(grid={"min_spacing_pct": "0.4"}, data_dir=tmp_path)


def test_yaml_and_env_layering(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "c.yaml"
    cfg.write_text("grid:\n  levels: 6\n  order_notional: 70000000\nmarket:\n  maker_fee: 0.002\n")
    monkeypatch.setenv("NBX_CONFIG", str(cfg))
    monkeypatch.setenv("NBX_GRID__LEVELS", "3")
    s = Settings()
    assert s.grid.levels == 3  # env beats yaml
    assert s.grid.order_notional == Decimal("70000000")  # yaml beats default
    assert s.market.maker_fee == Decimal("0.002")
