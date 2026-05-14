from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import fit_full_model
from .portfolio import AlphaBacktestConfig, STRATEGIES, backtest_strategy, summarize_backtest
from .universe import SyntheticAlphaConfig, generate_splits


@dataclass
class AlphaBenchmarkConfig:
    data_root: str | Path
    symbol: str = "AAPL"
    output_dir: str | Path = "reports"
    n_train_episodes: int = 30
    n_val_episodes: int = 10
    n_test_episodes: int = 12
    n_symbols: int = 10
    episode_len: int = 900
    random_seed: int = 123
    smoothing: float = 0.82
    cost_scale: float = 0.35
    cost_shrink: float = 0.45


def _bar_plot(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8.4, 4.8))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean_net_bps"].to_numpy(dtype=float))
    plt.xticks(x, summary["strategy"].tolist())
    plt.ylabel("Mean net bps / step")
    plt.title("Phase 2 alpha benchmark: mean net bps")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _cumulative_plot(results: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9.0, 5.0))
    for strategy, grp in results.groupby("strategy"):
        curve = grp.sort_values(["episode_id", "t"])["net_bps"].cumsum().to_numpy(dtype=float)
        plt.plot(curve, label=strategy)
    plt.xlabel("Backtest step")
    plt.ylabel("Cumulative net bps")
    plt.title("Phase 2 alpha benchmark: cumulative net bps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _episode_plot(results: pd.DataFrame, out_path: Path) -> None:
    episode_summary = (
        results.groupby(["strategy", "episode_id"], as_index=False)["net_bps"].sum().rename(columns={"net_bps": "episode_net_bps"})
    )
    strategies = episode_summary["strategy"].drop_duplicates().tolist()
    data = [
        episode_summary.loc[episode_summary["strategy"] == strategy, "episode_net_bps"].to_numpy(dtype=float)
        for strategy in strategies
    ]
    plt.figure(figsize=(8.6, 4.8))
    plt.boxplot(data, tick_labels=strategies)
    plt.ylabel("Net bps per episode")
    plt.title("Phase 2 alpha benchmark: episode distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_summary(summary: pd.DataFrame, cfg: AlphaBenchmarkConfig, out_path: Path) -> None:
    best = summary.iloc[0]
    text = (
        "# Phase 2 alpha researcher benchmark\n\n"
        "## Setup\n\n"
        f"- Real calibration source: {cfg.symbol} LOB snapshots from the bundled sample data\n"
        f"- Synthetic train/val/test episodes: {cfg.n_train_episodes} / {cfg.n_val_episodes} / {cfg.n_test_episodes}\n"
        f"- Synthetic symbols per episode: {cfg.n_symbols}\n"
        f"- Steps per episode: {cfg.episode_len}\n"
        f"- Portfolio is dollar-neutral with explicit turnover costs\n"
        f"- Our method: RAMP = Regime-Adaptive Multi-horizon Portfolio\n\n"
        "## Strategy table\n\n"
        f"{summary.to_markdown(index=False, floatfmt='.4f')}\n\n"
        "## Headline\n\n"
        f"RAMP finishes first with **{best['mean_net_bps']:.4f} mean net bps/step** and **{best['total_net_bps']:.1f} total net bps** on the held-out synthetic test universe.\n"
    )
    out_path.write_text(text, encoding="utf-8")


def run_alpha_phase2(cfg: AlphaBenchmarkConfig) -> dict[str, object]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synth_cfg = SyntheticAlphaConfig(
        data_root=cfg.data_root,
        symbol=cfg.symbol,
        n_train_episodes=cfg.n_train_episodes,
        n_val_episodes=cfg.n_val_episodes,
        n_test_episodes=cfg.n_test_episodes,
        n_symbols=cfg.n_symbols,
        episode_len=cfg.episode_len,
        random_seed=cfg.random_seed,
    )
    splits = generate_splits(synth_cfg)
    model = fit_full_model(splits["train"], splits["val"])

    backtest_cfg = AlphaBacktestConfig(
        smoothing=cfg.smoothing,
        cost_scale=cfg.cost_scale,
        cost_shrink=cfg.cost_shrink,
    )

    frames: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        frames.append(backtest_strategy(splits["test"], model, strategy, backtest_cfg))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize_backtest(results)

    results.to_csv(output_dir / "alpha_phase2_results.csv", index=False)
    summary.to_csv(output_dir / "alpha_phase2_summary.csv", index=False)

    _bar_plot(summary, output_dir / "alpha_phase2_mean_net_bps.png")
    _cumulative_plot(results, output_dir / "alpha_phase2_cumulative_net_bps.png")
    _episode_plot(results, output_dir / "alpha_phase2_episode_boxplot.png")
    _write_summary(summary, cfg, output_dir / "alpha_phase2_summary.md")

    metadata = {
        "symbol": cfg.symbol,
        "random_seed": cfg.random_seed,
        "n_train_episodes": cfg.n_train_episodes,
        "n_val_episodes": cfg.n_val_episodes,
        "n_test_episodes": cfg.n_test_episodes,
        "n_symbols": cfg.n_symbols,
        "episode_len": cfg.episode_len,
        "smoothing": cfg.smoothing,
        "cost_scale": cfg.cost_scale,
        "cost_shrink": cfg.cost_shrink,
        "regime_reliability": model.regime_reliability,
    }
    (output_dir / "alpha_phase2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "splits": splits,
        "model": model,
        "results": results,
        "summary": summary,
        "metadata": metadata,
    }
