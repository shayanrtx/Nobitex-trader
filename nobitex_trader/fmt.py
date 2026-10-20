"""Human-facing (Persian) formatting for the journal, notifications and dashboard."""

from __future__ import annotations

from decimal import Decimal

SIDE_FA = {"buy": "خرید", "sell": "فروش"}

RISK_FA = {
    "halted": "توقف ریسکی",
    "no_book": "اردربوک در دسترس نیست",
    "stale": "داده‌ی قیمت کهنه است",
    "one_sided": "اردربوک یک‌طرفه است",
    "spread": "اسپرد غیرعادی",
    "circuit": "حرکت شدید قیمت (توقف موقت)",
    "daily_loss": "حد ضرر روزانه — توقف تا فردا",
    "paused": "توقف دستی",
}

GUARD_FA = {
    "falling fast: buys paused": "ریزش سریع قیمت — خرید موقتاً متوقف شد",
    "rising fast: sells paused": "رشد سریع قیمت — فروش موقتاً متوقف شد",
}


def fmt_rial(value: Decimal) -> str:
    """Toman string from a rial amount: 1,055,550 rial → '105,555 تومان'."""
    return f"{int(value / 10):,} تومان"


def fmt_amount(value: Decimal) -> str:
    return format(value.normalize(), "f") if value == value.to_integral() else f"{value:f}"
