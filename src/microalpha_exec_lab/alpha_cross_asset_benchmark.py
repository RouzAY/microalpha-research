from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alpha_cross_asset_models import (
    CrossAssetStrategyConfig,
    StrategyContext,
    build_train_reference,
    fit_real_ridge_model,
    make_real_alpha_frame,
    prior_micro_sign,
    strategy_camp_r,
    strategy_micro_fixed,
    strategy_momentum,
    strategy_online_regime,
    strategy_ramp_real,
    strategy_ridge_static,
    strategy_signed,
    summarize_strategy,
)
from .multi_asset_data import (
    list_available_days_binance,
    list_available_days_optiver,
    load_binance_day,
    load_optiver_day,
)

STRATEGIES = [
    "momentum",
    "signed_flow",
    "micro_fixed",
    "ridge_static",
    "online_regime",
    "ramp_real",
    "camp_r",
]


@dataclass
class AlphaPhase4Config:
    optiver_root: str | Path
    binance_root: str | Path
    optiver_symbol: str = "AAPL"
    binance_symbol: str = "BNBUSDT"
    output_dir: str | Path = "reports"
    strategy: CrossAssetStrategyConfig = field(default_factory=CrossAssetStrategyConfig)
    eval_start_index: int = 1
    binance_sample_seconds: int = 1


@dataclass
class EpisodeSpec:
    asset: str
    venue: str
    day: str
    frame: pd.DataFrame
    train_frames: list[pd.DataFrame]
    evaluation_label: str



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



def _asset_bar_plot(asset_summary: pd.DataFrame, out_path: Path) -> None:
    pivot = asset_summary.pivot(index="strategy", columns="asset", values="mean_net_bps").sort_index()
    strategies = pivot.index.tolist()
    assets = pivot.columns.tolist()
    x = np.arange(len(strategies), dtype=float)
    width = 0.8 / max(1, len(assets))
    plt.figure(figsize=(9.4, 5.2))
    for idx, asset in enumerate(assets):
        vals = pivot[asset].fillna(0.0).to_numpy(dtype=float)
        plt.bar(x + idx * width - 0.4 + width / 2.0, vals, width=width, label=asset)
    plt.xticks(x, strategies, rotation=15)
    plt.ylabel("mean net bps")
    plt.title("Phase 4: mean net bps by asset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def _cumulative_plot(details: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9.4, 5.0))
    for strategy, grp in details.groupby("strategy"):
        grp = grp.sort_values(["asset", "day", "t"])
        curve = grp["net_bps"].cumsum().to_numpy(dtype=float)
        plt.plot(curve, label=strategy)
    plt.xlabel("Combined multi-asset backtest step")
    plt.ylabel("Cumulative net bps")
    plt.title("Phase 4: cumulative net bps across real multi-asset episodes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def _episode_plot(episode_summary: pd.DataFrame, out_path: Path) -> None:
    episode_summary = episode_summary.copy()
    episode_summary["episode"] = episode_summary["asset"] + "-" + episode_summary["day"]
    pivot = episode_summary.pivot(index="episode", columns="strategy", values="total_net_bps")
    plt.figure(figsize=(10.4, 5.4))
    for strategy in pivot.columns:
        plt.plot(pivot.index.astype(str), pivot[strategy].to_numpy(dtype=float), marker="o", label=strategy)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(rotation=35)
    plt.ylabel("Episode total net bps")
    plt.title("Phase 4: episode-level net bps across AAPL walk-forward + Binance external test")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def _write_summary(
    summary: pd.DataFrame,
    asset_summary: pd.DataFrame,
    episode_summary: pd.DataFrame,
    cfg: AlphaPhase4Config,
    out_path: Path,
    aapl_eval_days: list[str],
    bnb_days: list[str],
) -> None:
    best = summary.iloc[0]
    aapl_best = asset_summary[asset_summary["asset"].eq(cfg.optiver_symbol)].sort_values("mean_net_bps", ascending=False).iloc[0]
    bnb_best = asset_summary[asset_summary["asset"].eq(cfg.binance_symbol)].sort_values("mean_net_bps", ascending=False).iloc[0]
    text = (
        "# Phase 4 real multi-asset alpha benchmark\n\n"
        "## What phase 4 adds\n\n"
        "Phase 4 keeps the fully real AAPL walk-forward benchmark from phase 3 and adds a **second real asset / venue**: Binance **BNBUSDT**.\n"
        "The resulting benchmark is no longer just single-name time-series alpha. It now tests whether a microstructure method can remain useful under **cross-asset / cross-venue domain shift**.\n\n"
        "## Evaluation design\n\n"
        f"- In-domain track: {cfg.optiver_symbol} on XNAS, anchored walk-forward over days {', '.join(aapl_eval_days)}\n"
        f"- External track: {cfg.binance_symbol} on BINANCE, historical zero-shot external day(s): {', '.join(bnb_days)}\n"
        f"- Horizon / hold: {cfg.strategy.horizon} / {cfg.strategy.hold_steps}\n"
        f"- Warmup rows: {cfg.strategy.warmup}\n"
        f"- Threshold quantile: {cfg.strategy.threshold_quantile}\n"
        f"- Effective spread-cost coefficient: {cfg.strategy.cost_coeff}\n"
        f"- Domain-shift threshold for CAMP-R fallback: {cfg.strategy.shift_threshold}\n\n"
        "## Method\n\n"
        "Our new phase-4 method is **CAMP-R** = **Cross-Asset Adaptive Microstructure Portfolio on Real data**.\n\n"
        "CAMP-R works in two modes:\n"
        "1. **normal regime**: if the target episode looks statistically close to the training distribution, CAMP-R uses the stronger phase-3 hybrid (**RAMP-R** style micro + ML score);\n"
        "2. **domain-shift regime**: if the warmup distribution is far from the training universe, CAMP-R falls back to the safer microstructure-only regime logic.\n\n"
        "This is intentionally simple and recruiter-friendly: it clearly answers the practical question \"what should the model do when a new venue or asset looks nothing like the training set?\"\n\n"
        "## Combined multi-asset summary\n\n"
        f"{summary.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Per-asset summary\n\n"
        f"{asset_summary.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Episode-level totals\n\n"
        f"{episode_summary.to_markdown(index=False, floatfmt='.2f')}\n\n"
        "## Headline\n\n"
        f"Across the combined real multi-asset benchmark, **{best['strategy']}** finishes first with **{best['mean_net_bps']:.4f} mean net bps/step**, **{best['total_net_bps']:.1f} total net bps**, and **{int(best['positive_episodes'])}/{int(best['n_episodes'])} positive episodes**.\n\n"
        f"On in-domain {cfg.optiver_symbol}, the best strategy is **{aapl_best['strategy']}** at **{aapl_best['mean_net_bps']:.4f} mean net bps/step**.\n\n"
        f"On external {cfg.binance_symbol}, the best strategy is **{bnb_best['strategy']}** at **{bnb_best['mean_net_bps']:.4f} mean net bps/step**.\n\n"
        "## Honesty note\n\n"
        "This phase is **more realistic and more useful for recruiting** than phase 3 because it includes a second real asset and a second venue.\n"
        "But it is still a compact benchmark: there is only one external Binance day bundled in the repo, so the cross-asset claim is best presented as a **robustness / transfer test**, not as a fully diversified production universe.\n"
    )
    out_path.write_text(text, encoding="utf-8")



def _build_episode_specs(cfg: AlphaPhase4Config) -> tuple[list[EpisodeSpec], list[str], list[str]]:
    aapl_days = list_available_days_optiver(cfg.optiver_root, symbol=cfg.optiver_symbol)
    aapl_frames = {
        day: make_real_alpha_frame(load_optiver_day(cfg.optiver_root, day=day, symbol=cfg.optiver_symbol).book, horizon=cfg.strategy.horizon)
        for day in aapl_days
    }
    aapl_eval_days = aapl_days[cfg.eval_start_index :]

    specs: list[EpisodeSpec] = []
    for day in aapl_eval_days:
        train_days = [d for d in aapl_days if d < day]
        specs.append(
            EpisodeSpec(
                asset=cfg.optiver_symbol,
                venue="XNAS",
                day=day,
                frame=aapl_frames[day],
                train_frames=[aapl_frames[d] for d in train_days],
                evaluation_label="aapl_walkforward",
            )
        )

    bnb_days = list_available_days_binance(cfg.binance_root, symbol=cfg.binance_symbol)
    for day in bnb_days:
        bnb_frame = make_real_alpha_frame(
            load_binance_day(cfg.binance_root, day=day, symbol=cfg.binance_symbol, sample_seconds=cfg.binance_sample_seconds).book,
            horizon=cfg.strategy.horizon,
        )
        specs.append(
            EpisodeSpec(
                asset=cfg.binance_symbol,
                venue="BINANCE",
                day=day,
                frame=bnb_frame,
                train_frames=[aapl_frames[d] for d in aapl_days],
                evaluation_label="binance_external",
            )
        )

    return specs, aapl_eval_days, bnb_days



def run_alpha_phase4(cfg: AlphaPhase4Config) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs, aapl_eval_days, bnb_days = _build_episode_specs(cfg)

    details_frames: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, object]] = []
    for spec in specs:
        context = StrategyContext(
            prior_sign=prior_micro_sign(spec.train_frames),
            model=fit_real_ridge_model(spec.train_frames, cfg.strategy),
            train_reference=build_train_reference(spec.train_frames),
        )
        details_frames.extend(
            [
                strategy_momentum(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy),
                strategy_signed(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy),
                strategy_micro_fixed(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy, context.prior_sign),
                strategy_ridge_static(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy, context.model, context.prior_sign),
                strategy_online_regime(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy, context.prior_sign),
                strategy_ramp_real(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy, context.model, context.prior_sign),
                strategy_camp_r(spec.asset, spec.venue, spec.day, spec.frame, cfg.strategy, context),
            ]
        )
        schedule_rows.append(
            {
                "asset": spec.asset,
                "venue": spec.venue,
                "day": spec.day,
                "evaluation_label": spec.evaluation_label,
                "n_train_frames": len(spec.train_frames),
                "prior_sign": int(context.prior_sign),
            }
        )

    details = pd.concat(details_frames, ignore_index=True)

    summary_rows: list[dict[str, object]] = []
    episode_summaries: list[pd.DataFrame] = []
    for strategy, grp in details.groupby("strategy"):
        ep_summary, summary = summarize_strategy(grp)
        summary["strategy"] = strategy
        summary_rows.append(summary)
        episode_summaries.append(ep_summary)
    summary = pd.DataFrame(summary_rows).sort_values("mean_net_bps", ascending=False).reset_index(drop=True)
    episode_summary = pd.concat(episode_summaries, ignore_index=True).sort_values(["asset", "day", "strategy"]).reset_index(drop=True)

    asset_summary = (
        details.groupby(["asset", "venue", "strategy"], as_index=False)
        .agg(
            mean_net_bps=("net_bps", "mean"),
            total_net_bps=("net_bps", "sum"),
            mean_gross_bps=("gross_bps", "mean"),
            mean_cost_bps=("cost_bps", "mean"),
            turnover=("turnover", "mean"),
            active_share=("active", "mean"),
        )
        .sort_values(["asset", "mean_net_bps"], ascending=[True, False])
        .reset_index(drop=True)
    )

    external_summary = asset_summary[asset_summary["asset"].eq(cfg.binance_symbol)].copy().reset_index(drop=True)

    details.to_csv(output_dir / "alpha_phase4_results.csv", index=False)
    summary.to_csv(output_dir / "alpha_phase4_summary.csv", index=False)
    asset_summary.to_csv(output_dir / "alpha_phase4_asset_summary.csv", index=False)
    episode_summary.to_csv(output_dir / "alpha_phase4_episode_summary.csv", index=False)
    external_summary.to_csv(output_dir / "alpha_phase4_external_summary.csv", index=False)
    pd.DataFrame(schedule_rows).to_csv(output_dir / "alpha_phase4_schedule.csv", index=False)

    _bar_plot(summary, output_dir / "alpha_phase4_mean_net_bps.png", "Phase 4: mean net bps across real multi-asset benchmark")
    _asset_bar_plot(asset_summary, output_dir / "alpha_phase4_asset_mean_net_bps.png")
    _cumulative_plot(details, output_dir / "alpha_phase4_cumulative_net_bps.png")
    _episode_plot(episode_summary, output_dir / "alpha_phase4_episode_total_net_bps.png")
    _bar_plot(external_summary, output_dir / "alpha_phase4_external_mean_net_bps.png", "Phase 4: external Binance mean net bps")

    metadata = {
        "optiver_symbol": cfg.optiver_symbol,
        "binance_symbol": cfg.binance_symbol,
        "aapl_eval_days": aapl_eval_days,
        "binance_days": bnb_days,
        "strategies": STRATEGIES,
        "phase4_config": {**asdict(cfg.strategy), "eval_start_index": cfg.eval_start_index, "binance_sample_seconds": cfg.binance_sample_seconds},
        "best_overall": summary.iloc[0].to_dict(),
        "best_external": external_summary.iloc[0].to_dict(),
    }
    (output_dir / "alpha_phase4_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_summary(summary, asset_summary, episode_summary, cfg, output_dir / "alpha_phase4_summary.md", aapl_eval_days, bnb_days)

    return {
        "details": details,
        "summary": summary,
        "asset_summary": asset_summary,
        "episode_summary": episode_summary,
        "external_summary": external_summary,
        "metadata": metadata,
    }
