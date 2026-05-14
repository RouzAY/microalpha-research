from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

FEATURE_COLUMNS_REAL = [
    "spread",
    "imb1",
    "imb2",
    "micro_dev",
    "trade_signed_roll5",
    "trade_signed_roll15",
    "trade_abs_roll5",
    "trade_abs_roll15",
    "ntrades_roll5",
    "ret_5",
    "ret_15",
    "ret_30",
    "sig_micro",
    "sig_flow",
    "signed_imb5",
    "signed_imb15",
    "vol15",
    "ntr15",
    "spread_x_imb",
]


@dataclass
class RidgeRealModel:
    feature_columns: list[str]
    mean_: np.ndarray
    std_: np.ndarray
    coef_: np.ndarray
    ridge_lambda: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=float)
        xs = (x - self.mean_) / self.std_
        return xs @ self.coef_


@dataclass
class CrossAssetStrategyConfig:
    horizon: int = 30
    warmup: int = 800
    hold_steps: int = 30
    threshold_quantile: float = 0.85
    cost_coeff: float = 0.12
    ridge_lambda: float = 3000.0
    beta_switch_threshold: float = 2.5
    micro_weight: float = 0.7
    ml_weight: float = 0.3
    shift_threshold: float = 3.0


@dataclass
class TrainReference:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class StrategyContext:
    prior_sign: int
    model: RidgeRealModel
    train_reference: TrainReference


def make_real_alpha_frame(book: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    frame = book.copy()
    frame["spread_bps"] = frame["spread"] / frame["mid"] * 1e4
    frame["future_ret_bps"] = (frame["mid"].shift(-horizon) / frame["mid"] - 1.0) * 1e4
    frame["sig_micro"] = frame["micro_dev"] / (frame["spread"] + 1e-12)
    frame["sig_flow"] = frame["trade_signed_roll15"] / (frame["trade_abs_roll15"] + 1.0)
    frame["signed_imb5"] = frame["trade_signed_roll5"] / (frame["trade_abs_roll5"] + 1.0)
    frame["signed_imb15"] = frame["trade_signed_roll15"] / (frame["trade_abs_roll15"] + 1.0)
    frame["vol15"] = np.log1p(frame["trade_abs_roll15"])
    frame["ntr15"] = np.log1p(frame["ntrades_roll15"])
    frame["spread_x_imb"] = frame["spread_bps"] * frame["imb1"]
    return frame.dropna().reset_index(drop=True)


def fit_real_ridge_model(day_frames: Iterable[pd.DataFrame], cfg: CrossAssetStrategyConfig) -> RidgeRealModel:
    stacked = pd.concat(list(day_frames), ignore_index=True)
    x = stacked[FEATURE_COLUMNS_REAL].to_numpy(dtype=float)
    y = stacked["future_ret_bps"].to_numpy(dtype=float)
    mean_ = x.mean(axis=0)
    std_ = x.std(axis=0) + 1e-9
    xs = (x - mean_) / std_
    eye = np.eye(xs.shape[1], dtype=float)
    coef = np.linalg.solve(xs.T @ xs + cfg.ridge_lambda * eye, xs.T @ y)
    return RidgeRealModel(
        feature_columns=list(FEATURE_COLUMNS_REAL),
        mean_=mean_,
        std_=std_,
        coef_=coef,
        ridge_lambda=cfg.ridge_lambda,
    )


def build_train_reference(day_frames: Iterable[pd.DataFrame]) -> TrainReference:
    stacked = pd.concat(list(day_frames), ignore_index=True)
    mean = stacked[FEATURE_COLUMNS_REAL].mean().to_numpy(dtype=float)
    std = stacked[FEATURE_COLUMNS_REAL].std().to_numpy(dtype=float) + 1e-9
    return TrainReference(mean=mean, std=std)


def prior_micro_sign(train_frames: Iterable[pd.DataFrame]) -> int:
    corrs: list[float] = []
    for frame in train_frames:
        x = frame["sig_micro"].to_numpy(dtype=float)
        y = frame["future_ret_bps"].to_numpy(dtype=float)
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(corr):
            corrs.append(corr)
    if not corrs:
        return 1
    return 1 if float(np.mean(corrs)) >= 0.0 else -1


def _threshold(raw: np.ndarray, warmup: int, q: float) -> float:
    warm = max(32, min(warmup, len(raw)))
    return float(np.quantile(np.abs(raw[:warm]), q))


def _make_positions(raw: np.ndarray, threshold: float, hold_steps: int, sign: int, warmup: int) -> np.ndarray:
    n = len(raw)
    warm = max(32, min(warmup, n))
    pos = np.zeros(n, dtype=float)
    for t in range(warm, n, hold_steps):
        score = float(raw[t])
        p = 0.0 if abs(score) < threshold else float(np.sign(sign * score))
        pos[t : min(n, t + hold_steps)] = p
    return pos


def _regime_beta(frame: pd.DataFrame, warmup: int) -> float:
    warm = max(32, min(warmup, len(frame)))
    x = frame["sig_micro"].to_numpy(dtype=float)[:warm]
    y = frame["future_ret_bps"].to_numpy(dtype=float)[:warm]
    return float(np.cov(x, y, bias=True)[0, 1] / (np.var(x) + 1e-12))


def domain_shift(frame: pd.DataFrame, ref: TrainReference, warmup: int) -> float:
    warm = max(32, min(warmup, len(frame)))
    x = frame[FEATURE_COLUMNS_REAL].iloc[:warm].to_numpy(dtype=float)
    z = (x - ref.mean) / ref.std
    return float(np.sqrt(np.mean(z ** 2)))


def _backtest_from_positions(
    frame: pd.DataFrame,
    positions: np.ndarray,
    strategy: str,
    day: str,
    asset: str,
    venue: str,
    score: np.ndarray,
    cfg: CrossAssetStrategyConfig,
    regime_sign: int | None = None,
    regime_beta: float | None = None,
    shift_value: float | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    prev = np.r_[0.0, positions[:-1]]
    gross = positions * frame["future_ret_bps"].to_numpy(dtype=float)
    cost = cfg.cost_coeff * frame["spread_bps"].to_numpy(dtype=float) * np.abs(positions - prev)
    net = gross - cost
    out = pd.DataFrame(
        {
            "asset": asset,
            "venue": venue,
            "day": day,
            "strategy": strategy,
            "t": np.arange(len(frame), dtype=int),
            "future_ret_bps": frame["future_ret_bps"].to_numpy(dtype=float),
            "gross_bps": gross,
            "cost_bps": cost,
            "net_bps": net,
            "position": positions,
            "turnover": np.abs(positions - prev),
            "score": score,
            "spread_bps": frame["spread_bps"].to_numpy(dtype=float),
            "active": np.abs(positions) > 0.0,
        }
    )
    if regime_sign is not None:
        out["regime_sign"] = int(regime_sign)
    if regime_beta is not None:
        out["regime_beta"] = float(regime_beta)
    if shift_value is not None:
        out["domain_shift"] = float(shift_value)
    if mode is not None:
        out["mode"] = mode
    return out


def strategy_momentum(asset: str, venue: str, day: str, frame: pd.DataFrame, cfg: CrossAssetStrategyConfig) -> pd.DataFrame:
    raw = frame["ret_30"].to_numpy(dtype=float)
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=1, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "momentum", day, asset, venue, raw, cfg)


def strategy_signed(asset: str, venue: str, day: str, frame: pd.DataFrame, cfg: CrossAssetStrategyConfig) -> pd.DataFrame:
    raw = frame["sig_flow"].to_numpy(dtype=float)
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=1, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "signed_flow", day, asset, venue, raw, cfg)


def strategy_micro_fixed(
    asset: str,
    venue: str,
    day: str,
    frame: pd.DataFrame,
    cfg: CrossAssetStrategyConfig,
    prior_sign: int,
) -> pd.DataFrame:
    raw = frame["sig_micro"].to_numpy(dtype=float)
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=prior_sign, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "micro_fixed", day, asset, venue, raw, cfg, regime_sign=prior_sign)


def strategy_ridge_static(
    asset: str,
    venue: str,
    day: str,
    frame: pd.DataFrame,
    cfg: CrossAssetStrategyConfig,
    model: RidgeRealModel,
    prior_sign: int,
) -> pd.DataFrame:
    raw = model.predict(frame)
    warm = max(32, min(cfg.warmup, len(raw)))
    raw = (raw - np.median(raw[:warm])) / (np.std(raw[:warm]) + 1e-9)
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=prior_sign, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "ridge_static", day, asset, venue, raw, cfg, regime_sign=prior_sign)


def strategy_online_regime(
    asset: str,
    venue: str,
    day: str,
    frame: pd.DataFrame,
    cfg: CrossAssetStrategyConfig,
    prior_sign: int,
) -> pd.DataFrame:
    raw = frame["sig_micro"].to_numpy(dtype=float)
    beta = _regime_beta(frame, cfg.warmup)
    morning_sign = 1 if beta >= 0.0 else -1
    regime_sign = morning_sign if abs(beta) >= cfg.beta_switch_threshold else prior_sign
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=regime_sign, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "online_regime", day, asset, venue, raw, cfg, regime_sign=regime_sign, regime_beta=beta)


def strategy_ramp_real(
    asset: str,
    venue: str,
    day: str,
    frame: pd.DataFrame,
    cfg: CrossAssetStrategyConfig,
    model: RidgeRealModel,
    prior_sign: int,
) -> pd.DataFrame:
    raw_micro = frame["sig_micro"].to_numpy(dtype=float)
    raw_ml = model.predict(frame)
    warm = max(32, min(cfg.warmup, len(raw_ml)))
    raw_ml = (raw_ml - np.median(raw_ml[:warm])) / (np.std(raw_ml[:warm]) + 1e-9)
    raw = cfg.micro_weight * raw_micro + cfg.ml_weight * raw_ml
    beta = _regime_beta(frame, cfg.warmup)
    morning_sign = 1 if beta >= 0.0 else -1
    regime_sign = morning_sign if abs(beta) >= cfg.beta_switch_threshold else prior_sign
    threshold = _threshold(raw, warmup=cfg.warmup, q=cfg.threshold_quantile)
    pos = _make_positions(raw, threshold=threshold, hold_steps=cfg.hold_steps, sign=regime_sign, warmup=cfg.warmup)
    return _backtest_from_positions(frame, pos, "ramp_real", day, asset, venue, raw, cfg, regime_sign=regime_sign, regime_beta=beta)


def strategy_camp_r(
    asset: str,
    venue: str,
    day: str,
    frame: pd.DataFrame,
    cfg: CrossAssetStrategyConfig,
    context: StrategyContext,
) -> pd.DataFrame:
    shift_value = domain_shift(frame, context.train_reference, warmup=cfg.warmup)
    if shift_value > cfg.shift_threshold:
        out = strategy_online_regime(asset, venue, day, frame, cfg, context.prior_sign).copy()
        out["strategy"] = "camp_r"
        out["domain_shift"] = float(shift_value)
        out["mode"] = "micro_fallback"
        return out

    out = strategy_ramp_real(asset, venue, day, frame, cfg, context.model, context.prior_sign).copy()
    out["strategy"] = "camp_r"
    out["domain_shift"] = float(shift_value)
    out["mode"] = "ramp"
    return out


def summarize_strategy(details: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int]]:
    episode_summary = (
        details.groupby(["strategy", "asset", "venue", "day"], as_index=False)
        .agg(
            total_net_bps=("net_bps", "sum"),
            mean_net_bps=("net_bps", "mean"),
            mean_gross_bps=("gross_bps", "mean"),
            mean_cost_bps=("cost_bps", "mean"),
            turnover=("turnover", "mean"),
            active_share=("active", "mean"),
            hit_rate=("net_bps", lambda x: float(np.mean(np.asarray(x) > 0.0))),
        )
        .sort_values(["strategy", "asset", "day"])
        .reset_index(drop=True)
    )
    episode_totals = episode_summary["total_net_bps"].to_numpy(dtype=float)
    tstat = float(
        episode_totals.mean() / (episode_totals.std(ddof=1) + 1e-9) * np.sqrt(len(episode_totals))
    ) if len(episode_totals) > 1 else float("nan")
    summary = {
        "mean_net_bps": float(details["net_bps"].mean()),
        "total_net_bps": float(details["net_bps"].sum()),
        "mean_gross_bps": float(details["gross_bps"].mean()),
        "mean_cost_bps": float(details["cost_bps"].mean()),
        "turnover": float(details["turnover"].mean()),
        "active_share": float(details["active"].mean()),
        "hit_rate": float(np.mean(details["net_bps"].to_numpy(dtype=float) > 0.0)),
        "n_episodes": int(episode_summary[["asset", "day"]].drop_duplicates().shape[0]),
        "positive_episodes": int((episode_summary["total_net_bps"] > 0.0).sum()),
        "tstat_episode_total": tstat,
    }
    return episode_summary, summary
