from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..data import FEATURE_COLUMNS, list_available_days, load_many_days

HORIZONS = [1, 2, 5, 10, 20]
FAST_HORIZONS = [1, 2, 5]
SLOW_HORIZONS = [10, 20]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


@dataclass
class SyntheticAlphaConfig:
    data_root: str | Path
    symbol: str = "AAPL"
    train_days: list[str] | None = None
    val_days: list[str] | None = None
    test_days: list[str] | None = None
    n_train_episodes: int = 30
    n_val_episodes: int = 10
    n_test_episodes: int = 12
    n_symbols: int = 10
    episode_len: int = 900
    random_seed: int = 123


@dataclass
class SyntheticEpisode:
    episode_id: int
    symbols: list[str]
    features: dict[str, np.ndarray]
    regimes: dict[str, np.ndarray]
    realized_returns: dict[str, np.ndarray]
    horizon_returns: dict[str, dict[int, np.ndarray]]
    cost_bps: dict[str, np.ndarray]

    @property
    def n_steps(self) -> int:
        return len(self.realized_returns[self.symbols[0]])


class FeaturePool:
    def __init__(self, frame: pd.DataFrame) -> None:
        cols = FEATURE_COLUMNS + ["trade_abs", "mid"]
        self.frame = frame[cols].dropna().reset_index(drop=True)
        self.mean_ = self.frame[FEATURE_COLUMNS].mean()
        self.std_ = self.frame[FEATURE_COLUMNS].std() + 1e-8

    def sample_block(self, rng: np.random.Generator, length: int) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        total = 0
        while total < length:
            start = int(rng.integers(0, len(self.frame) - 160))
            block_len = int(rng.integers(50, 160))
            block = self.frame.iloc[start : start + block_len]
            parts.append(block)
            total += len(block)
        return pd.concat(parts, ignore_index=True).iloc[:length].copy().reset_index(drop=True)

    def standardize(self, frame: pd.DataFrame) -> np.ndarray:
        return ((frame[FEATURE_COLUMNS] - self.mean_) / self.std_).to_numpy(dtype=float)


def default_day_split(days: list[str]) -> tuple[list[str], list[str], list[str]]:
    if len(days) < 10:
        raise ValueError("At least 10 real days are required to build the phase-2 benchmark")
    return days[:6], days[6:8], days[8:10]


def build_feature_pool(data_root: str | Path, symbol: str, train_days: Iterable[str]) -> FeaturePool:
    loaded = load_many_days(data_root=data_root, days=list(train_days), symbol=symbol)
    frame = pd.concat([loaded[day].book for day in train_days], ignore_index=True)
    return FeaturePool(frame)


def observed_regime(frame: pd.DataFrame) -> np.ndarray:
    spread_z = (frame["spread"] - frame["spread"].median()) / (frame["spread"].std() + 1e-8)
    vol_proxy = frame["ret_5"].abs()
    vol_z = (vol_proxy - vol_proxy.median()) / (vol_proxy.std() + 1e-8)
    micro_proxy = frame["micro_dev"].abs()
    micro_z = (micro_proxy - micro_proxy.median()) / (micro_proxy.std() + 1e-8)

    regime = np.where(
        (vol_z > 0.5) & (micro_z > 0.5),
        2,
        np.where(
            (spread_z > 0.4) & (vol_z > 0.2),
            3,
            np.where(frame["ret_30"] > frame["ret_30"].median(), 1, 0),
        ),
    )
    return regime.astype(int)


def beta_library() -> dict[int, dict[int, np.ndarray]]:
    betas: dict[int, dict[int, np.ndarray]] = {}
    idx = FEATURE_INDEX
    d = len(FEATURE_COLUMNS)

    for regime in range(4):
        betas[regime] = {}
        for horizon in HORIZONS:
            b = np.zeros(d, dtype=float)
            if horizon <= 2:
                b[idx["imb1"]] = 1.2
                b[idx["micro_dev"]] = 1.5
                b[idx["trade_signed_roll5"]] = 1.0
                b[idx["ret_5"]] = -0.9
                b[idx["ret_30"]] = -0.1
            elif horizon == 5:
                b[idx["imb1"]] = 1.0
                b[idx["micro_dev"]] = 1.2
                b[idx["trade_signed_roll5"]] = 0.9
                b[idx["ret_5"]] = -0.6
                b[idx["ret_30"]] = 0.2
            else:
                b[idx["imb1"]] = 0.3
                b[idx["micro_dev"]] = 0.4
                b[idx["trade_signed_roll15"]] = 0.9
                b[idx["ret_5"]] = 0.3
                b[idx["ret_30"]] = 1.3

            if regime == 0:
                if horizon >= 10:
                    b *= 0.5
                else:
                    b[idx["ret_5"]] -= 0.4
            elif regime == 1:
                if horizon <= 5:
                    b *= 0.65
                else:
                    b[idx["ret_30"]] += 0.8
                    b[idx["trade_signed_roll15"]] += 0.4
            elif regime == 2:
                if horizon >= 10:
                    b *= 0.35
                else:
                    b[idx["micro_dev"]] += 0.8
                    b[idx["imb1"]] += 0.6
            elif regime == 3:
                b *= 0.18
            betas[regime][horizon] = b
    return betas


def generate_episode(
    pool: FeaturePool,
    episode_id: int,
    n_symbols: int,
    episode_len: int,
    seed: int,
) -> SyntheticEpisode:
    rng = np.random.default_rng(seed)
    betas = beta_library()

    factor = np.zeros(episode_len, dtype=float)
    eps = rng.normal(0.0, 0.09, size=episode_len)
    for t in range(1, episode_len):
        factor[t] = 0.9 * factor[t - 1] + eps[t]

    symbols = [f"S{idx:02d}" for idx in range(n_symbols)]
    features: dict[str, np.ndarray] = {}
    regimes: dict[str, np.ndarray] = {}
    realized_returns: dict[str, np.ndarray] = {}
    horizon_returns: dict[str, dict[int, np.ndarray]] = {}
    cost_bps: dict[str, np.ndarray] = {}

    for symbol in symbols:
        sampled = pool.sample_block(rng, episode_len)
        x = pool.standardize(sampled)
        regime = observed_regime(sampled)
        style = rng.normal(1.0, 0.15, size=(len(HORIZONS), len(FEATURE_COLUMNS)))

        spread_bps = sampled["spread"].to_numpy(dtype=float) / sampled["mid"].to_numpy(dtype=float) * 1e4
        liq = np.log1p(sampled["trade_abs"].to_numpy(dtype=float))
        vol = np.abs(sampled["ret_5"].to_numpy(dtype=float)) * 1e4
        symbol_cost = 0.025 * spread_bps + 0.03 / (liq + 0.5) + 0.008 * vol + 0.03
        noise_scale = 0.20 + 0.018 * spread_bps

        rets: dict[int, np.ndarray] = {}
        for h_idx, horizon in enumerate(HORIZONS):
            mean_bps = np.zeros(episode_len, dtype=float)
            for t in range(episode_len):
                mean_bps[t] = (x[t] @ (betas[int(regime[t])][horizon] * style[h_idx])) * 0.42
            mean_bps += factor * (0.08 if horizon <= 5 else 0.18)
            rets[horizon] = (mean_bps + rng.normal(0.0, noise_scale, size=episode_len)) / 1e4

        gate_fast = 1.0 / (
            1.0
            + np.exp(
                -(
                    1.1 * x[:, FEATURE_INDEX["micro_dev"]]
                    - 0.9 * x[:, FEATURE_INDEX["ret_30"]]
                    - 0.5 * x[:, FEATURE_INDEX["spread"]]
                )
            )
        )
        realized = gate_fast * (0.65 * rets[2] + 0.35 * rets[5]) + (1.0 - gate_fast) * (
            0.55 * rets[10] + 0.45 * rets[20]
        )

        features[symbol] = x
        regimes[symbol] = regime
        realized_returns[symbol] = realized
        horizon_returns[symbol] = rets
        cost_bps[symbol] = symbol_cost

    return SyntheticEpisode(
        episode_id=episode_id,
        symbols=symbols,
        features=features,
        regimes=regimes,
        realized_returns=realized_returns,
        horizon_returns=horizon_returns,
        cost_bps=cost_bps,
    )


def generate_splits(cfg: SyntheticAlphaConfig) -> dict[str, list[SyntheticEpisode]]:
    days = list_available_days(cfg.data_root, symbol=cfg.symbol)
    if cfg.train_days is None or cfg.val_days is None or cfg.test_days is None:
        cfg.train_days, cfg.val_days, cfg.test_days = default_day_split(days)

    pool = build_feature_pool(cfg.data_root, cfg.symbol, cfg.train_days)
    base_seed = int(cfg.random_seed)

    train = [
        generate_episode(pool, episode_id=i, n_symbols=cfg.n_symbols, episode_len=cfg.episode_len, seed=base_seed + 100 + i)
        for i in range(cfg.n_train_episodes)
    ]
    val = [
        generate_episode(pool, episode_id=i, n_symbols=cfg.n_symbols, episode_len=cfg.episode_len, seed=base_seed + 200 + i)
        for i in range(cfg.n_val_episodes)
    ]
    test = [
        generate_episode(pool, episode_id=i, n_symbols=cfg.n_symbols, episode_len=cfg.episode_len, seed=base_seed + 300 + i)
        for i in range(cfg.n_test_episodes)
    ]
    return {"train": train, "val": val, "test": test}
