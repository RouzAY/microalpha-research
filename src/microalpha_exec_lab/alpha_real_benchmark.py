from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alpha_real_models import (
    RealStrategyConfig,
    fit_real_ridge_model,
    make_real_alpha_frame,
    prior_micro_sign,
    strategy_micro_fixed,
    strategy_momentum,
    strategy_online_regime,
    strategy_ramp_real,
    strategy_ridge_static,
    strategy_signed,
    summarize_strategy,
)
from .data import list_available_days, load_day

STRATEGIES = [
    "momentum",
    "signed_flow",
    "micro_fixed",
    "ridge_static",
    "online_regime",
    "ramp_real",
]

@dataclass
class AlphaPhase3Config:
    data_root: str | Path
    symbol: str = "AAPL"
    output_dir: str | Path = "reports"
    strategy: RealStrategyConfig = field(default_factory=RealStrategyConfig)
    eval_start_index: int = 1
    heldout_days: int = 2


def _bar_plot(summary: pd.DataFrame, out_path: Path, title: str, value_col: str = "mean_net_bps") -> None:
    ordered = summary.sort_values(value_col, ascending=False).reset_index(drop=True)
    plt.figure(figsize=(8.8, 4.8))
    x = np.arange(len(ordered))
    plt.bar(x, ordered[value_col].to_numpy(dtype=float))
    plt.xticks(x, ordered["strategy"].tolist(), rotation=15)
    plt.ylabel(value_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _cumulative_plot(details: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9.2, 5.0))
    for strategy, grp in details.groupby("strategy"):
        curve = grp.sort_values(["day", "t"])["net_bps"].cumsum().to_numpy(dtype=float)
        plt.plot(curve, label=strategy)
    plt.xlabel("Walk-forward backtest step")
    plt.ylabel("Cumulative net bps")
    plt.title("Phase 3: cumulative net bps on real walk-forward evaluation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _daily_plot(day_summary: pd.DataFrame, out_path: Path) -> None:
    pivot = day_summary.pivot(index="day", columns="strategy", values="total_net_bps")
    plt.figure(figsize=(9.4, 5.2))
    for strategy in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[strategy].to_numpy(dtype=float), marker="o", label=strategy)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(rotation=35)
    plt.ylabel("Daily total net bps")
    plt.title("Phase 3: daily out-of-sample net bps by strategy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_summary(summary: pd.DataFrame, heldout_summary: pd.DataFrame, day_summary: pd.DataFrame, cfg: AlphaPhase3Config, out_path: Path, eval_days: list[str], heldout_days: list[str]) -> None:
    best = summary.iloc[0]
    best_hold = heldout_summary.iloc[0]
    text = (
        "# Phase 3 fully-real walk-forward alpha benchmark\n\n"
        "## What changes in phase 3\n\n"
        "Phase 3 removes the synthetic multi-symbol universe and evaluates everything on **real AAPL limit-order-book days only**.\n"
        "The benchmark is now a strict **anchored walk-forward** setup: each evaluation day is traded using only previous days plus a same-day warmup window.\n\n"
        "## Method\n\n"
        "Our phase-3 method is **RAMP-R** = **Regime-Adaptive Microstructure Portfolio on Real data**.\n\n"
        "RAMP-R combines:\n"
        "1. a microstructure signal based on microprice displacement / imbalance;\n"
        "2. a ridge model trained on all prior real days;\n"
        "3. an online morning regime test to decide whether the classic imbalance signal should keep or flip sign;\n"
        "4. sparse trading with a fixed holding period;\n"
        "5. an effective execution-cost model proportional to spread × turnover.\n\n"
        "## Evaluation window\n\n"
        f"- Symbol: {cfg.symbol}\n"
        f"- Real days available: {len(eval_days) + cfg.eval_start_index}\n"
        f"- Walk-forward evaluation days: {', '.join(eval_days)}\n"
        f"- Final held-out days: {', '.join(heldout_days)}\n"
        f"- Horizon / hold: {cfg.strategy.horizon} / {cfg.strategy.hold_steps}\n"
        f"- Warmup rows: {cfg.strategy.warmup}\n"
        f"- Threshold quantile: {cfg.strategy.threshold_quantile}\n"
        f"- Effective cost coefficient: {cfg.strategy.cost_coeff}\n\n"
        "## Full walk-forward summary\n\n"
        f"{summary.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Final held-out summary\n\n"
        f"{heldout_summary.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Daily totals\n\n"
        f"{day_summary.to_markdown(index=False, floatfmt='.2f')}\n\n"
        "## Headline\n\n"
        f"Across all real out-of-sample walk-forward days, **{best['strategy']}** finishes first with **{best['mean_net_bps']:.4f} mean net bps/step**, **{best['total_net_bps']:.1f} total net bps**, and **{int(best['positive_days'])}/{int(best['n_days'])} positive days**.\n\n"
        f"On the strict final held-out block, **{best_hold['strategy']}** is first at **{best_hold['mean_net_bps']:.4f} mean net bps/step**.\n\n"
        "## Honesty note\n\n"
        "This phase is **fully real in the sense of data and walk-forward evaluation**, but it is still limited by the bundled sample: only one symbol and only ten days are available.\n"
        "So this is a strong *research-platform* result, not a claim of production-ready, multi-asset live alpha.\n"
    )
    out_path.write_text(text, encoding="utf-8")


def run_alpha_phase3(cfg: AlphaPhase3Config) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    days = list_available_days(cfg.data_root, symbol=cfg.symbol)
    books = {day: load_day(cfg.data_root, day=day, symbol=cfg.symbol).book for day in days}
    frames = {day: make_real_alpha_frame(book, horizon=cfg.strategy.horizon) for day, book in books.items()}
    eval_days = days[cfg.eval_start_index :]
    heldout_days = eval_days[-cfg.heldout_days :]

    details_frames: list[pd.DataFrame] = []
    walkforward_rows: list[dict[str, object]] = []
    for day in eval_days:
        train_days = [d for d in days if d < day]
        train_frames = [frames[d] for d in train_days]
        prior_sign = prior_micro_sign(train_frames)
        ridge_model = fit_real_ridge_model(train_frames, cfg.strategy)
        frame = frames[day]
        strategy_outputs = [
            strategy_momentum(day, frame, cfg.strategy),
            strategy_signed(day, frame, cfg.strategy),
            strategy_micro_fixed(day, frame, cfg.strategy, prior_sign=prior_sign),
            strategy_ridge_static(day, frame, cfg.strategy, model=ridge_model, prior_sign=prior_sign),
            strategy_online_regime(day, frame, cfg.strategy, prior_sign=prior_sign),
            strategy_ramp_real(day, frame, cfg.strategy, model=ridge_model, prior_sign=prior_sign),
        ]
        details_frames.extend(strategy_outputs)
        walkforward_rows.append({"day": day, "n_train_days": len(train_days), "prior_sign": int(prior_sign), "train_days": ", ".join(train_days)})

    details = pd.concat(details_frames, ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    day_summaries: list[pd.DataFrame] = []
    for strategy, grp in details.groupby("strategy"):
        day_summary, summary = summarize_strategy(grp)
        summary["strategy"] = strategy
        summary_rows.append(summary)
        day_summaries.append(day_summary)
    summary = pd.DataFrame(summary_rows).sort_values("mean_net_bps", ascending=False).reset_index(drop=True)
    day_summary = pd.concat(day_summaries, ignore_index=True).sort_values(["day", "strategy"]).reset_index(drop=True)

    heldout = details[details["day"].isin(heldout_days)].copy()
    heldout_rows: list[dict[str, object]] = []
    for strategy, grp in heldout.groupby("strategy"):
        _, hs = summarize_strategy(grp)
        hs["strategy"] = strategy
        heldout_rows.append(hs)
    heldout_summary = pd.DataFrame(heldout_rows).sort_values("mean_net_bps", ascending=False).reset_index(drop=True)

    details.to_csv(output_dir / "alpha_phase3_results.csv", index=False)
    summary.to_csv(output_dir / "alpha_phase3_summary.csv", index=False)
    day_summary.to_csv(output_dir / "alpha_phase3_daily_summary.csv", index=False)
    heldout_summary.to_csv(output_dir / "alpha_phase3_heldout_summary.csv", index=False)
    pd.DataFrame(walkforward_rows).to_csv(output_dir / "alpha_phase3_walkforward_schedule.csv", index=False)

    _bar_plot(summary, output_dir / "alpha_phase3_mean_net_bps.png", "Phase 3: mean net bps on real walk-forward evaluation")
    _cumulative_plot(details, output_dir / "alpha_phase3_cumulative_net_bps.png")
    _daily_plot(day_summary, output_dir / "alpha_phase3_daily_total_net_bps.png")
    _bar_plot(heldout_summary, output_dir / "alpha_phase3_heldout_mean_net_bps.png", "Phase 3: mean net bps on final held-out days")

    metadata = {
        "symbol": cfg.symbol,
        "all_days": days,
        "eval_days": eval_days,
        "heldout_days": heldout_days,
        "strategies": STRATEGIES,
        "phase3_config": {**asdict(cfg.strategy), "eval_start_index": cfg.eval_start_index, "heldout_days": cfg.heldout_days},
        "best_full": summary.iloc[0].to_dict(),
        "best_heldout": heldout_summary.iloc[0].to_dict(),
    }
    (output_dir / "alpha_phase3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_summary(summary, heldout_summary, day_summary, cfg, output_dir / "alpha_phase3_summary.md", eval_days, heldout_days)

    return {"details": details, "summary": summary, "day_summary": day_summary, "heldout_summary": heldout_summary, "metadata": metadata}
