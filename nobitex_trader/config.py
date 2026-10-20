"""Typed configuration.

Non-secret settings live in a YAML file (``config.yaml``); secrets come from the
environment / ``.env``. Any YAML value can be overridden from the environment with
``NBX_<SECTION>__<KEY>``, e.g. ``NBX_GRID__LEVELS=6``.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from nobitex_trader.domain.market import MarketSpec


class ExchangeSettings(BaseModel):
    base_url: str = "https://apiv2.nobitex.ir"
    ws_url: str = "wss://ws.nobitex.ir/connection/websocket"
    # Nobitex asks bots to identify as TraderBot/<name>.
    user_agent: str = "TraderBot/NobitexTrader"
    timeout_seconds: float = 10.0
    use_websocket: bool = True


class MarketSettings(BaseModel):
    symbol: str = "USDTIRT"
    src: str = "usdt"
    dst: str = "rls"
    price_tick: Decimal = Decimal("10")
    amount_step: Decimal = Decimal("0.01")
    # Exchange minimum order value: 3,000,000 rial on IRT markets, 11 USDT on USDT markets.
    min_notional: Decimal = Decimal("3000000")
    # Set these to YOUR fee tier (Nobitex profile → fees). Used for spacing & PnL.
    maker_fee: Decimal = Decimal("0.0025")
    taker_fee: Decimal = Decimal("0.0025")

    def spec(self) -> MarketSpec:
        return MarketSpec(
            symbol=self.symbol.upper(),
            src=self.src.lower(),
            dst=self.dst.lower(),
            price_tick=self.price_tick,
            amount_step=self.amount_step,
            min_notional=self.min_notional,
            maker_fee=self.maker_fee,
            taker_fee=self.taker_fee,
        )


class CapitalSettings(BaseModel):
    """The slice of the account the bot is allowed to manage.

    The bot never touches funds beyond these caps, even if the wallet holds more.
    """

    max_quote: Decimal = Field(Decimal("500000000"), description="rial the bot may use")
    max_base: Decimal = Field(Decimal("0"), description="USDT the bot may use at start")


class GridSettings(BaseModel):
    """Adaptive, inventory-aware grid around a volatility-scaled fair price."""

    candle_resolution: str = "15"
    candle_history: int = 200
    fair_ema_period: int = 48
    atr_period: int = 14
    trend_fast_ema: int = 24
    trend_slow_ema: int = 96

    levels: int = 4
    order_notional: Decimal = Field(Decimal("50000000"), description="rial per grid order")
    spacing_atr_mult: Decimal = Decimal("0.6")
    min_spacing_pct: Decimal = Decimal("0.7")
    max_spacing_pct: Decimal = Decimal("3.0")
    level_growth: Decimal = Decimal("1.25")  # each further level is spaced wider

    target_base_ratio: Decimal = Decimal("0.5")
    min_base_ratio: Decimal = Decimal("0.15")
    max_base_ratio: Decimal = Decimal("0.85")
    trend_skew: Decimal = Decimal("0.25")  # how far trend moves the inventory target
    inventory_skew: Decimal = Decimal("1.0")  # how hard inventory imbalance shifts quotes

    # Fast signals (react within seconds, not candles):
    # order-book imbalance over the top N levels tilts the grid centre toward pressure.
    imbalance_levels: int = 5
    imbalance_weight: Decimal = Decimal("0.3")
    # Momentum guard: if price moved more than X spacings within the window, stop
    # quoting against the move (no knife-catching buys / no selling into a pump).
    momentum_window_seconds: float = 180.0
    momentum_threshold_spacing: Decimal = Decimal("0.8")

    # Classic-grid discipline: never sell below average cost + sell fee + this margin.
    # Suits USDT/IRT (long-term upward drift): in dips the bot accumulates up to
    # max_base_ratio and waits instead of realising losses.
    sell_above_cost: bool = True
    min_profit_over_cost_pct: Decimal = Decimal("0.2")

    @model_validator(mode="after")
    def _check(self) -> GridSettings:
        if not (Decimal(0) <= self.min_base_ratio <= self.target_base_ratio
                <= self.max_base_ratio <= Decimal(1)):
            raise ValueError(
                "require 0 <= min_base_ratio <= target_base_ratio <= max_base_ratio <= 1"
            )
        if self.min_spacing_pct <= 0 or self.max_spacing_pct < self.min_spacing_pct:
            raise ValueError("invalid spacing bounds")
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        return self


class RiskSettings(BaseModel):
    max_daily_loss_pct: Decimal = Decimal("3")
    max_drawdown_pct: Decimal = Decimal("10")
    max_spread_pct: Decimal = Decimal("1.5")
    stale_data_seconds: float = 30.0
    circuit_breaker_move_pct: Decimal = Decimal("2.5")
    circuit_breaker_window_seconds: float = 120.0
    circuit_breaker_cooldown_seconds: float = 600.0
    max_open_orders: int = 12
    # Nobitex rejects prices > 30% away from market; stay well inside.
    max_price_deviation_pct: Decimal = Decimal("10")
    timezone: str = "Asia/Tehran"


class EngineSettings(BaseModel):
    tick_seconds: float = 5.0
    min_requote_seconds: float = 3.0
    requote_tolerance_pct: Decimal = Decimal("0.15")
    order_sync_seconds: float = 6.0
    equity_snapshot_seconds: float = 60.0
    cancel_on_shutdown: bool = True
    client_order_prefix: str = "nbxt"
    # Nobitex shared order limit is 300 / 10 min; keep headroom.
    order_rate_limit: int = 240
    order_rate_window_seconds: float = 600.0


class PaperSettings(BaseModel):
    initial_base: Decimal = Decimal("0")
    initial_quote: Decimal = Decimal("500000000")


class NotifySettings(BaseModel):
    enabled: bool = False
    # Telegram-compatible bot API. Use https://tapi.bale.ai for Bale (works inside Iran).
    api_base: str = "https://api.telegram.org"
    chat_id: str = ""
    proxy: str | None = None  # e.g. socks5://127.0.0.1:1080 — only used for notifications
    notify_fills: bool = True


class DashboardSettings(BaseModel):
    enabled: bool = True
    # 127.0.0.1 keeps it private to this machine. Inside Docker use 0.0.0.0 and publish
    # the port only on 127.0.0.1 (see docker-compose.yml).
    host: str = "127.0.0.1"
    port: int = 8080


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NBX_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    mode: Literal["paper", "live"] = "paper"
    strategy: str = "adaptive_grid"
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    log_json: bool = False

    # -- secrets (env only) -------------------------------------------------------
    api_token: SecretStr | None = None  # "Authorization: Token ..." style
    api_key: str | None = None  # Nobitex-Key (API key, Ed25519)
    api_private_key: SecretStr | None = None
    notify_bot_token: SecretStr | None = None
    dashboard_token: SecretStr | None = None  # required for dashboard access

    exchange: ExchangeSettings = ExchangeSettings()
    market: MarketSettings = MarketSettings()
    capital: CapitalSettings = CapitalSettings()
    grid: GridSettings = GridSettings()
    risk: RiskSettings = RiskSettings()
    engine: EngineSettings = EngineSettings()
    paper: PaperSettings = PaperSettings()
    notify: NotifySettings = NotifySettings()
    dashboard: DashboardSettings = DashboardSettings()

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check_live(self) -> Settings:
        if self.mode == "live" and not (self.api_token or (self.api_key and self.api_private_key)):
            raise ValueError(
                "live mode requires NBX_API_TOKEN or NBX_API_KEY + NBX_API_PRIVATE_KEY"
            )
        spacing_floor = self.market.maker_fee * 2 * 100
        if self.grid.min_spacing_pct <= spacing_floor:
            raise ValueError(
                f"grid.min_spacing_pct ({self.grid.min_spacing_pct}%) must exceed the round-trip "
                f"maker fee ({spacing_floor}%) or every grid cycle loses money"
            )
        return self

    @property
    def db_path(self) -> Path:
        return self.data_dir / f"trader-{self.mode}.sqlite3"

    @property
    def heartbeat_path(self) -> Path:
        return self.data_dir / "heartbeat"

    # YAML sits *below* env vars and init kwargs in priority.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        path = Path(os.environ.get("NBX_CONFIG", "config.yaml"))
        self._data: dict[str, Any] = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path} must contain a mapping at top level")
            self._data = loaded

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


def load_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
