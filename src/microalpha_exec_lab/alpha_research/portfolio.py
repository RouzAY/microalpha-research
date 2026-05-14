from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .models import RidgeEnsemble
from .universe import FAST_HORIZONS, FEATURE_INDEX, HORIZONS, SLOW_HORIZONS, SyntheticEpisode

StrategyName = Literal["imbalance", "momentum", "ridge5", "uniform", "ramp"]
STRATEGIES = ["imbalance", "momentum", "ridge5", "uniform", "ramp"]


@dataclass
class AlphaBacktestConfig:
    smoothing: float = 0.82
    cost_scale: float = 0.35
    cost_shrink: float = 0.45


@dataclass
class StrategyMetrics:
    strategy: str
    mean_gross_bps: float
    mean_net_bps: float
    total_net_bps: float
    hit_rate: float
    mean_turnover: float
    sharpe_like: float


def _cross_sectional_weights(
    score: np.ndarray,
    risk_bps: np.ndarray,
    cost_bps: np.ndarray,
    ours: bool,
    cost_shrink: float,
) -> np.ndarray:
    signal = score / (risk_bps + 2.0)
    if ours:
        signal = signal / (1.0 + cost_shrink * cost_bps)
    signal = signal - signal.mean()
    signal = np.tanh(1.8 * signal)
    gross = np.sum(np.abs(signal))
    if gross <= 1e-12:
        return np.zeros_like(signal)
    return signal / gross


def _uniform_score(model: RidgeEnsemble, episode: SyntheticEpisode, symbols: list[str], t: int) -> np.ndarray:
    mat = np.column_stack(
        [
            np.array([episode.features[symbol][t] @ model.coef_by_horizon[h] for symbol in symbols], dtype=float)
            for h in HORIZONS
        ]
    )
    mat = (mat - np.median(mat, axis=0)) / (np.std(mat, axis=0) + 1e-8)
    return mat.mean(axis=1)


def _ramp_score(
    model: RidgeEnsemble,
    episode: SyntheticEpisode,
    symbols: list[str],
    t: int,
    prev_smooth: np.ndarray,
) -> np.ndarray:
    raw = np.zeros(len(symbols), dtype=float)
    for i, symbol in enumerate(symbols):
        regime = int(episode.regimes[symbol][t])
        rel = model.regime_reliability[regime]

        w_fast = np.array([rel[h] for h in FAST_HORIZONS], dtype=float)
        w_slow = np.array([rel[h] for h in SLOW_HORIZONS], dtype=float)
        if w_fast.sum() <= 0.0:
            w_fast = np.ones_like(w_fast)
        if w_slow.sum() <= 0.0:
            w_slow = np.ones_like(w_slow)
        w_fast = w_fast / w_fast.sum()
        w_slow = w_slow / w_slow.sum()

        p_fast = np.array([episode.features[symbol][t] @ model.coef_by_horizon[h] for h in FAST_HORIZONS], dtype=float)
        p_slow = np.array([episode.features[symbol][t] @ model.coef_by_horizon[h] for h in SLOW_HORIZONS], dtype=float)
        s_fast = float((w_fast * p_fast).sum())
        s_slow = float((w_slow * p_slow).sum())

        x = episode.features[symbol][t]
        gate_fast = 1.0 / (
            1.0
            + np.exp(
                -(
                    1.1 * x[FEATURE_INDEX["micro_dev"]]
                    - 0.9 * x[FEATURE_INDEX["ret_30"]]
                    - 0.5 * x[FEATURE_INDEX["spread"]]
                )
            )
        )
        raw[i] = gate_fast * s_fast + (1.0 - gate_fast) * s_slow
    return raw


def backtest_strategy(
    episodes: list[SyntheticEpisode],
    model: RidgeEnsemble,
    strategy: StrategyName,
    cfg: AlphaBacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for episode in episodes:
        symbols = sorted(episode.symbols)
        prev_weights = np.zeros(len(symbols), dtype=float)
        smooth_score = np.zeros(len(symbols), dtype=float)

        for t in range(episode.n_steps):
            cost_bps = np.array([episode.cost_bps[symbol][t] for symbol in symbols], dtype=float)
            risk_bps = np.array(
                [np.std(episode.realized_returns[symbol][max(0, t - 30) : t + 1]) * 1e4 for symbol in symbols],
                dtype=float,
            )
            risk_bps = risk_bps + 0.3

            if strategy == "imbalance":
                score = np.array([episode.features[symbol][t, FEATURE_INDEX["imb1"]] for symbol in symbols], dtype=float)
                weights = _cross_sectional_weights(score, risk_bps, cost_bps, ours=False, cost_shrink=cfg.cost_shrink)
            elif strategy == "momentum":
                score = np.array([episode.features[symbol][t, FEATURE_INDEX["ret_30"]] for symbol in symbols], dtype=float)
                weights = _cross_sectional_weights(score, risk_bps, cost_bps, ours=False, cost_shrink=cfg.cost_shrink)
            elif strategy == "ridge5":
                score = np.array([episode.features[symbol][t] @ model.coef_by_horizon[5] for symbol in symbols], dtype=float)
                weights = _cross_sectional_weights(score, risk_bps, cost_bps, ours=False, cost_shrink=cfg.cost_shrink)
            elif strategy == "uniform":
                score = _uniform_score(model, episode, symbols, t)
                weights = _cross_sectional_weights(score, risk_bps, cost_bps, ours=False, cost_shrink=cfg.cost_shrink)
            elif strategy == "ramp":
                score = _ramp_score(model, episode, symbols, t, prev_smooth=smooth_score)
                smooth_score = cfg.smoothing * smooth_score + (1.0 - cfg.smoothing) * score
                weights = _cross_sectional_weights(
                    smooth_score,
                    risk_bps,
                    cost_bps,
                    ours=True,
                    cost_shrink=cfg.cost_shrink,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            realized = np.array([episode.realized_returns[symbol][t] for symbol in symbols], dtype=float)
            gross_bps = float((prev_weights * realized).sum() * 1e4)
            turnover = float(np.abs(weights - prev_weights).sum())
            trading_cost = float((cost_bps * np.abs(weights - prev_weights)).sum() * cfg.cost_scale)
            net_bps = gross_bps - trading_cost

            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "t": t,
                    "strategy": strategy,
                    "gross_bps": gross_bps,
                    "net_bps": net_bps,
                    "turnover": turnover,
                }
            )
            prev_weights = weights

    return pd.DataFrame(rows)


def summarize_backtest(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("strategy", as_index=False).agg(
        mean_gross_bps=("gross_bps", "mean"),
        mean_net_bps=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
        hit_rate=("net_bps", lambda x: float((x > 0).mean())),
        mean_turnover=("turnover", "mean"),
        std_net_bps=("net_bps", "std"),
    )
    grouped["sharpe_like"] = grouped["mean_net_bps"] / (grouped["std_net_bps"] + 1e-12)
    grouped = grouped.drop(columns=["std_net_bps"]).sort_values("mean_net_bps", ascending=False).reset_index(drop=True)
    return grouped
