from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load_many_days, list_available_days
from .model import RidgeSideModel, fit_side_model
from .policy import POLICIES, PolicyConfig, simulate_parent_order
from .synthetic import BootstrapGenerator


@dataclass
class BenchmarkConfig:
    data_root: str | Path
    symbol: str = "AAPL"
    train_days: list[str] | None = None
    val_days: list[str] | None = None
    test_days: list[str] | None = None
    qtys: tuple[int, ...] = (1000, 3000, 5000)
    window_len: int = 900
    warmup: int = 600
    stride: int = 1800
    model_horizon: int = 20
    ridge_lambda: float = 100.0
    n_synth_episodes: int = 50
    bootstrap_block_size: int = 60
    random_seed: int = 42
    output_dir: str | Path = "reports"


def _default_split(days: list[str]) -> tuple[list[str], list[str], list[str]]:
    if len(days) < 10:
        raise ValueError("Default split expects at least 10 days")
    return days[:6], days[6:8], days[8:10]


def _volume_profile(day_frames: list[pd.DataFrame]) -> np.ndarray:
    max_len = max(len(frame) for frame in day_frames)
    stacked: list[np.ndarray] = []
    for frame in day_frames:
        x = frame["trade_abs"].to_numpy(dtype=float)
        if len(x) < max_len:
            x = np.pad(x, (0, max_len - len(x)))
        stacked.append(x)
    prof = np.mean(stacked, axis=0) + 1e-6
    return prof / prof.sum()


def _window_slices(frame: pd.DataFrame, warmup: int, window_len: int, stride: int) -> list[tuple[int, int]]:
    starts = list(range(warmup, len(frame) - 2 * stride, stride))
    return [(start, start + window_len) for start in starts if start + window_len < len(frame)]


def _run_real_eval(
    test_frames: dict[str, pd.DataFrame],
    qtys: tuple[int, ...],
    model: RidgeSideModel,
    train_profile: np.ndarray,
    config: PolicyConfig,
    warmup: int,
    window_len: int,
    stride: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for day, frame in test_frames.items():
        for start, end in _window_slices(frame, warmup=warmup, window_len=window_len, stride=stride):
            window = frame.iloc[start : end + 1].reset_index(drop=True)
            vol_prof = train_profile[start : end + 1]
            vol_prof = vol_prof / vol_prof.sum()
            for qty in qtys:
                for side in ["BUY", "SELL"]:
                    for policy in POLICIES:
                        result = simulate_parent_order(
                            frame=window,
                            side=side,
                            qty=qty,
                            policy=policy,
                            model=model,
                            volume_profile=vol_prof,
                            config=config,
                        )
                        rows.append(
                            {
                                "dataset": "real",
                                "day": day,
                                "start_idx": start,
                                "end_idx": end,
                                "qty": qty,
                                "side": side,
                                "policy": policy,
                                "is_bps": result.is_bps,
                            }
                        )
    return pd.DataFrame(rows)


def _run_synth_eval(
    train_frames: list[pd.DataFrame],
    qtys: tuple[int, ...],
    model: RidgeSideModel,
    config: PolicyConfig,
    n_episodes: int,
    block_size: int,
    episode_len: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    generator = BootstrapGenerator(train_frames=train_frames, random_seed=random_seed)
    episodes = generator.make_many(n_episodes=n_episodes, block_size=block_size, episode_len=episode_len)
    equal_profile = np.ones(episode_len, dtype=float) / episode_len
    for episode_id, frame in enumerate(episodes):
        frame = frame.reset_index(drop=True)
        for qty in qtys:
            for side in ["BUY", "SELL"]:
                for policy in POLICIES:
                    result = simulate_parent_order(
                        frame=frame,
                        side=side,
                        qty=qty,
                        policy=policy,
                        model=model,
                        volume_profile=equal_profile,
                        config=config,
                    )
                    rows.append(
                        {
                            "dataset": "synthetic",
                            "episode_id": episode_id,
                            "qty": qty,
                            "side": side,
                            "policy": policy,
                            "is_bps": result.is_bps,
                        }
                    )
    return pd.DataFrame(rows)


def _select_policy_weights(
    val_frames: dict[str, pd.DataFrame],
    model: RidgeSideModel,
    train_profile: np.ndarray,
    qty: int,
    warmup: int,
    window_len: int,
    stride: int,
) -> PolicyConfig:
    best_score: tuple[float, float, float, float] | None = None
    for liq_weight in np.linspace(0.4, 1.0, 7):
        for alpha_weight in np.linspace(0.0, 0.8, 9):
            config = PolicyConfig(liq_weight=float(liq_weight), alpha_weight=float(alpha_weight))
            vals: list[float] = []
            for _, frame in val_frames.items():
                for start, end in _window_slices(frame, warmup=warmup, window_len=window_len, stride=stride):
                    window = frame.iloc[start : end + 1].reset_index(drop=True)
                    vol_prof = train_profile[start : end + 1]
                    vol_prof = vol_prof / vol_prof.sum()
                    for side in ["BUY", "SELL"]:
                        result = simulate_parent_order(
                            frame=window,
                            side=side,
                            qty=qty,
                            policy="malp",
                            model=model,
                            volume_profile=vol_prof,
                            config=config,
                        )
                        vals.append(result.is_bps)
            mean_score = float(np.mean(vals))
            median_score = float(np.median(vals))
            candidate = (mean_score, median_score, float(liq_weight), float(alpha_weight))
            if best_score is None or candidate < best_score:
                best_score = candidate
    assert best_score is not None
    return PolicyConfig(liq_weight=best_score[2], alpha_weight=best_score[3])


def _make_barplot(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.8))
    x = np.arange(len(summary))
    plt.bar(x, summary["mean_is_bps"].to_numpy(dtype=float))
    plt.xticks(x, summary["policy"].tolist())
    plt.ylabel("Mean implementation shortfall (bps)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_markdown_summary(
    real_summary: pd.DataFrame,
    synth_summary: pd.DataFrame,
    cfg: BenchmarkConfig,
    policy_cfg: PolicyConfig,
    out_path: Path,
) -> None:
    text = (
        "# Benchmark summary\n\n"
        "## Experimental setup\n\n"
        f"- Symbol: {cfg.symbol}\n"
        f"- Train days: {', '.join(cfg.train_days or [])}\n"
        f"- Validation days: {', '.join(cfg.val_days or [])}\n"
        f"- Test days: {', '.join(cfg.test_days or [])}\n"
        f"- Parent order sizes: {', '.join(map(str, cfg.qtys))}\n"
        f"- Window length: {cfg.window_len} book updates\n"
        f"- MALP weights selected on validation set: liquidity={policy_cfg.liq_weight:.3f}, alpha={policy_cfg.alpha_weight:.3f}\n\n"
        "## Held-out real-data results\n\n"
        f"{real_summary.to_markdown(index=False, floatfmt='.3f')}\n\n"
        "## Synthetic stress-test results\n\n"
        f"{synth_summary.to_markdown(index=False, floatfmt='.3f')}\n"
    )
    out_path.write_text(text, encoding="utf-8")


def run_full_benchmark(cfg: BenchmarkConfig) -> dict[str, object]:
    data_root = Path(cfg.data_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_days = list_available_days(data_root, symbol=cfg.symbol)
    if cfg.train_days is None or cfg.val_days is None or cfg.test_days is None:
        cfg.train_days, cfg.val_days, cfg.test_days = _default_split(all_days)

    loaded = load_many_days(
        data_root=data_root,
        days=cfg.train_days + cfg.val_days + cfg.test_days,
        symbol=cfg.symbol,
    )
    train_frames = [loaded[day].book for day in cfg.train_days]
    val_frames = {day: loaded[day].book for day in cfg.val_days}
    test_frames = {day: loaded[day].book for day in cfg.test_days}

    model = fit_side_model(train_frames, horizon=cfg.model_horizon, ridge_lambda=cfg.ridge_lambda)
    model.save(output_dir / "malp_model.json")
    train_profile = _volume_profile(train_frames)

    policy_cfg = _select_policy_weights(
        val_frames=val_frames,
        model=model,
        train_profile=train_profile,
        qty=3000,
        warmup=cfg.warmup,
        window_len=cfg.window_len,
        stride=cfg.stride,
    )

    real_df = _run_real_eval(
        test_frames=test_frames,
        qtys=cfg.qtys,
        model=model,
        train_profile=train_profile,
        config=policy_cfg,
        warmup=cfg.warmup,
        window_len=cfg.window_len,
        stride=cfg.stride,
    )
    synth_df = _run_synth_eval(
        train_frames=train_frames,
        qtys=cfg.qtys,
        model=model,
        config=policy_cfg,
        n_episodes=cfg.n_synth_episodes,
        block_size=cfg.bootstrap_block_size,
        episode_len=cfg.window_len,
        random_seed=cfg.random_seed,
    )

    real_df.to_csv(output_dir / "real_results.csv", index=False)
    synth_df.to_csv(output_dir / "synthetic_results.csv", index=False)

    real_summary = real_df.groupby(["qty", "policy"], as_index=False)["is_bps"].mean().rename(
        columns={"is_bps": "mean_is_bps"}
    )
    synth_summary = synth_df.groupby(["qty", "policy"], as_index=False)["is_bps"].mean().rename(
        columns={"is_bps": "mean_is_bps"}
    )
    real_summary.to_csv(output_dir / "real_summary.csv", index=False)
    synth_summary.to_csv(output_dir / "synthetic_summary.csv", index=False)

    real_3k = real_summary.loc[real_summary["qty"] == 3000, ["policy", "mean_is_bps"]].sort_values("mean_is_bps")
    synth_3k = synth_summary.loc[synth_summary["qty"] == 3000, ["policy", "mean_is_bps"]].sort_values("mean_is_bps")
    _make_barplot(real_3k, output_dir / "real_mean_is_bps.png", "Held-out real data (3k shares)")
    _make_barplot(synth_3k, output_dir / "synthetic_mean_is_bps.png", "Synthetic bootstrap stress test (3k shares)")
    _write_markdown_summary(real_summary, synth_summary, cfg, policy_cfg, output_dir / "benchmark_summary.md")

    metadata = {
        "train_days": cfg.train_days,
        "val_days": cfg.val_days,
        "test_days": cfg.test_days,
        "policy_config": {
            "liq_weight": policy_cfg.liq_weight,
            "alpha_weight": policy_cfg.alpha_weight,
        },
        "qtys": list(cfg.qtys),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "real_results": real_df,
        "synthetic_results": synth_df,
        "real_summary": real_summary,
        "synthetic_summary": synth_summary,
        "policy_config": policy_cfg,
        "model": model,
        "metadata": metadata,
    }
