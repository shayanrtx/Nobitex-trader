"""Walk-forward parameter optimisation with overfitting diagnostics.

Method (following Pardo / Bailey & López de Prado practice):

1. Sample a *small* number of candidate configurations (default 40 — more trials
   mean more chance of fitting noise) from a coarse parameter grid, always
   including the current config.
2. Backtest every candidate once over the full history (in parallel) with the
   realistic fill model, recording an hourly equity/inventory curve.
3. Roll windows: in-sample (IS, default 21 days) then out-of-sample (OOS, 7 days),
   stepping by the OOS length. In each IS window pick a config by a robust rule —
   among configs within 10% of the best IS score, take the one whose *worst*
   IS week is best (prefer plateaus to peaks) — then record how that choice did
   on the following OOS window it has never seen.
4. Report the concatenated OOS performance, walk-forward efficiency, how often
   the IS winner ranked in the bottom half OOS (a simple PBO estimate), and the
   configuration that was most robust across all OOS windows.

Scores are *excess return over holding the window's starting inventory*, divided
by (1 + max drawdown %), scaled down when a config trades too little.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import random
import statistics
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from nobitex_trader.backtest import run_backtest
from nobitex_trader.config import Settings
from nobitex_trader.domain.models import Candle
from nobitex_trader.strategy.registry import create_strategy

Curve = list[tuple[int, float, float, float, float]]

# Coarse grid: few values per parameter, spanning sensible ranges for USDT/IRT.
DEFAULT_SPACE: dict[str, list[Any]] = {
    "min_spacing_pct": ["0.6", "0.8", "1.0", "1.3"],
    "spacing_atr_mult": ["0.4", "0.6", "0.9"],
    "levels": [3, 4, 6],
    "level_growth": ["1.0", "1.25", "1.5"],
    "trend_skew": ["0", "0.25", "0.4"],
    "inventory_skew": ["0.5", "1.0", "1.5"],
    "min_profit_over_cost_pct": ["0.1", "0.2", "0.4"],
    "regime_er_enter": ["0.4", "0.5", "0.65"],
    "momentum_threshold_spacing": ["0.6", "0.8", "1.2"],
}


@dataclass(slots=True)
class WindowStats:
    ret_pct: float
    excess_pct: float
    max_dd_pct: float
    fills: int
    score: float


def window_stats(curve: Curve, fills_ts: Sequence[int], start: int, end: int,
                 min_fills_per_day: float = 2.0) -> WindowStats | None:
    pts = [p for p in curve if start <= p[0] < end]
    if len(pts) < 24:
        return None
    t0, e0, base0, quote0, _ = pts[0]
    e1, price1 = pts[-1][1], pts[-1][4]
    hold = quote0 + base0 * price1
    ret = (e1 / e0 - 1) * 100
    excess = (e1 / hold - 1) * 100 if hold > 0 else 0.0
    peak, dd = e0, 0.0
    for p in pts:
        peak = max(peak, p[1])
        dd = max(dd, (peak - p[1]) / peak * 100)
    n_fills = sum(1 for t in fills_ts if start <= t < end)
    days = max((pts[-1][0] - t0) / 86400, 1e-9)
    activity = min(1.0, n_fills / (min_fills_per_day * days))
    score = excess / (1 + dd) * (activity if excess > 0 else 1.0)
    return WindowStats(ret, excess, dd, n_fills, score)


def sample_configs(space: dict[str, list[Any]], n: int, seed: int = 1) -> list[dict[str, Any]]:
    keys = sorted(space)
    total = math.prod(len(space[k]) for k in keys)
    rng = random.Random(seed)
    if total <= n:
        combos = list(itertools.product(*(space[k] for k in keys)))
    else:
        seen: set[tuple[Any, ...]] = set()
        combos = []
        while len(combos) < n:
            c = tuple(rng.choice(space[k]) for k in keys)
            if c not in seen:
                seen.add(c)
                combos.append(c)
    return [dict(zip(keys, c, strict=True)) for c in combos]


def _run_one(args: tuple[dict[str, Any], dict[str, Any], list[Candle], str]) -> dict[str, Any]:
    base_settings, overrides, candles, spread = args
    data = json.loads(json.dumps(base_settings, default=str))
    data["grid"].update({k: str(v) if not isinstance(v, int) else v for k, v in overrides.items()})
    data["dashboard"]["enabled"] = False
    settings = Settings(**data)
    result = asyncio.run(run_backtest(settings, create_strategy(settings), candles,
                                      Decimal(spread)))
    return {
        "overrides": overrides,
        "curve": result.curve,
        "fills_ts": result.fill_times,
        "summary": {
            "return_pct": float(result.return_pct),
            "excess_pct": float(result.excess_return_pct),
            "max_dd_pct": float(result.max_drawdown_pct),
            "fills": result.fills,
            "orders_placed": result.orders_placed,
            "markout_bps": result.markout_bps,
        },
    }


@dataclass(slots=True)
class FoldResult:
    is_start: int
    oos_start: int
    oos_end: int
    chosen: int
    oos: WindowStats
    is_score: float
    oos_rank_pct: float  # 0 = best OOS, 1 = worst


@dataclass(slots=True)
class OptimizationReport:
    configs: list[dict[str, Any]]
    folds: list[FoldResult]
    oos_median_scores: list[float]
    full_period: list[dict[str, Any]]

    @property
    def recommended(self) -> int:
        return max(range(len(self.configs)), key=lambda i: self.oos_median_scores[i])

    def summary(self) -> str:
        lines = [f"configs tested: {len(self.configs)}   walk-forward folds: {len(self.folds)}"]
        if not self.folds:
            return "\n".join([*lines, "not enough data for walk-forward (need IS + OOS days)"])
        lines.append("")
        lines.append("fold  OOS window            chosen  OOS excess  OOS maxDD  fills  OOS rank")
        for i, f in enumerate(self.folds):
            from datetime import UTC, datetime

            d0 = datetime.fromtimestamp(f.oos_start, UTC).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(f.oos_end, UTC).strftime("%Y-%m-%d")
            lines.append(f"{i + 1:>4}  {d0}..{d1}  #{f.chosen:<5} {f.oos.excess_pct:>+9.2f}% "
                         f"{f.oos.max_dd_pct:>8.2f}% {f.oos.fills:>6}  {f.oos_rank_pct:>7.0%}")
        oos_total = math.prod(1 + f.oos.excess_pct / 100 for f in self.folds) - 1
        positive = sum(1 for f in self.folds if f.oos.excess_pct > 0)
        is_mean = statistics.mean(f.is_score for f in self.folds)
        oos_mean = statistics.mean(f.oos.score for f in self.folds)
        wfe = oos_mean / is_mean if is_mean > 0 else float("nan")
        pbo = sum(1 for f in self.folds if f.oos_rank_pct > 0.5) / len(self.folds)
        lines += [
            "",
            f"walk-forward OOS excess return (compounded): {oos_total * 100:+.2f}%",
            f"profitable OOS windows: {positive}/{len(self.folds)}",
            f"walk-forward efficiency (OOS/IS score): {wfe:.2f}   (> 0.5 is commonly acceptable)",
            f"PBO estimate (IS winner in OOS bottom half): {pbo:.0%}   (< 30% is reassuring)",
            "",
            "most robust config across OOS windows (median OOS score):",
        ]
        rec = self.recommended
        full = self.full_period[rec]
        lines.append(f"  #{rec}  median OOS score {self.oos_median_scores[rec]:.3f}  "
                     f"full-period excess {full['excess_pct']:+.2f}%  "
                     f"maxDD {full['max_dd_pct']:.2f}%  fills {full['fills']}")
        lines.append("")
        lines.append("paste into config.yaml under grid:")
        for k, v in sorted(self.configs[rec].items()):
            lines.append(f"  {k}: {v}")
        if pbo > 0.3 or (not math.isnan(wfe) and wfe < 0.5):
            lines.append("")
            lines.append("⚠️  overfitting risk is high: treat these parameters with caution, "
                         "prefer paper-trading validation and wider/default values.")
        return "\n".join(lines)


def run_optimization(
    settings: Settings,
    candles: Sequence[Candle],
    space: dict[str, list[Any]] | None = None,
    n_configs: int = 40,
    is_days: float = 21,
    oos_days: float = 7,
    workers: int | None = None,
    spread: str = "0.05",
    seed: int = 1,
) -> OptimizationReport:
    space = space or DEFAULT_SPACE
    def norm(v: Any) -> Any:
        return v if isinstance(v, int) else str(v)

    current = {k: norm(getattr(settings.grid, k)) for k in space}
    sampled = [c for c in sample_configs(space, n_configs, seed)
               if {k: norm(v) for k, v in c.items()} != current]
    configs = [current, *sampled[: max(n_configs - 1, 0)]]
    base = settings.model_dump(mode="json")
    tasks = [(base, cfg, list(candles), spread) for cfg in configs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(_run_one, tasks))

    start, end = candles[0].ts, candles[-1].ts
    is_len, oos_len = int(is_days * 86400), int(oos_days * 86400)
    folds: list[FoldResult] = []
    oos_scores: list[list[float]] = [[] for _ in configs]
    t = start + 86400  # skip the first day (indicator warm-up)
    while t + is_len + oos_len <= end + 3600:
        is_s, oos_s, oos_e = t, t + is_len, t + is_len + oos_len
        is_stats = [window_stats(r["curve"], r["fills_ts"], is_s, oos_s) for r in runs]
        oos_stats = [window_stats(r["curve"], r["fills_ts"], oos_s, oos_e) for r in runs]
        valid = [i for i, s in enumerate(is_stats) if s is not None and oos_stats[i] is not None]
        if not valid:
            break
        best = max(is_stats[i].score for i in valid)  # type: ignore[union-attr]
        near = [i for i in valid
                if is_stats[i].score >= best - 0.1 * abs(best)]  # type: ignore[union-attr]

        def worst_week(i: int, is_s: int = is_s, oos_s: int = oos_s) -> float:
            weeks = []
            w = is_s
            while w < oos_s:
                st = window_stats(runs[i]["curve"], runs[i]["fills_ts"], w, min(w + 7 * 86400,
                                                                             oos_s))
                if st is not None:
                    weeks.append(st.score)
                w += 7 * 86400
            return min(weeks) if weeks else -math.inf

        chosen = max(near, key=worst_week)
        ranked = sorted(valid, key=lambda i: oos_stats[i].score, reverse=True)  # type: ignore[union-attr]
        rank_pct = ranked.index(chosen) / max(len(ranked) - 1, 1)
        for i in valid:
            oos_scores[i].append(oos_stats[i].score)  # type: ignore[union-attr]
        folds.append(FoldResult(is_s, oos_s, oos_e, chosen, oos_stats[chosen],  # type: ignore[arg-type]
                                is_stats[chosen].score, rank_pct))  # type: ignore[union-attr]
        t += oos_len

    medians = [statistics.median(s) if s else -math.inf for s in oos_scores]
    return OptimizationReport(configs, folds, medians, [r["summary"] for r in runs])
