from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .model import RidgeSideModel

PolicyName = Literal["twap", "vwap", "liq", "alpha", "malp"]
POLICIES = ["twap", "vwap", "liq", "alpha", "malp"]


@dataclass
class ExecResult:
    policy: str
    side: str
    qty: float
    avg_px: float
    arrival_mid: float
    end_mid: float
    is_bps: float


@dataclass
class PolicyConfig:
    liq_weight: float = 1.0
    alpha_weight: float = 0.8
    max_score_clip: float = 1.5


def standardized(x: np.ndarray) -> np.ndarray:
    return (x - np.median(x)) / (np.std(x) + 1e-8)


def liquidity_score(frame: pd.DataFrame, side: str) -> np.ndarray:
    side = side.upper()
    if side == "BUY":
        raw = np.log(frame["size_ask_1"].to_numpy(dtype=float) + frame["size_ask_2"].to_numpy(dtype=float))
    else:
        raw = np.log(frame["size_bid_1"].to_numpy(dtype=float) + frame["size_bid_2"].to_numpy(dtype=float))
    raw = raw - np.log(frame["spread"].to_numpy(dtype=float) + 1e-6)
    return standardized(raw)


def execution_price(row: pd.Series, qty: float, side: str) -> float:
    qty = max(float(qty), 1e-12)
    side = side.upper()
    spread = max(float(row["spread"]), 1e-6)

    if side == "BUY":
        px_1 = float(row["price_ask_1"])
        px_2 = float(row["price_ask_2"])
        size_1 = float(row["size_ask_1"])
        size_2 = float(row["size_ask_2"])
        px_3 = px_2 + spread
    else:
        px_1 = float(row["price_bid_1"])
        px_2 = float(row["price_bid_2"])
        size_1 = float(row["size_bid_1"])
        size_2 = float(row["size_bid_2"])
        px_3 = px_2 - spread

    fill_1 = min(qty, size_1)
    rem = qty - fill_1
    fill_2 = min(rem, size_2)
    rem_2 = rem - fill_2
    return (fill_1 * px_1 + fill_2 * px_2 + rem_2 * px_3) / qty


def schedule_weights(
    frame: pd.DataFrame,
    side: str,
    policy: PolicyName,
    model: RidgeSideModel | None,
    volume_profile: np.ndarray | None,
    config: PolicyConfig,
) -> np.ndarray:
    n = len(frame)
    if policy == "twap":
        w = np.ones(n, dtype=float)
    elif policy == "vwap":
        if volume_profile is None or len(volume_profile) != n:
            raise ValueError("VWAP policy requires a matching volume profile")
        w = np.asarray(volume_profile, dtype=float)
    elif policy == "liq":
        score = liquidity_score(frame, side)
        w = np.exp(np.clip(score, -config.max_score_clip, config.max_score_clip))
    elif policy == "alpha":
        if model is None:
            raise ValueError("Alpha policy requires a fitted model")
        score = standardized(model.predict(frame, side=side))
        w = np.exp(np.clip(score, -config.max_score_clip, config.max_score_clip))
    elif policy == "malp":
        if model is None:
            raise ValueError("MALP policy requires a fitted model")
        liq = liquidity_score(frame, side)
        alpha = standardized(model.predict(frame, side=side))
        score = config.liq_weight * liq + config.alpha_weight * alpha
        w = np.exp(np.clip(score, -config.max_score_clip, config.max_score_clip))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    w = np.asarray(w, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return np.ones(n, dtype=float) / n
    return w / w_sum


def simulate_parent_order(
    frame: pd.DataFrame,
    side: str,
    qty: float,
    policy: PolicyName,
    model: RidgeSideModel | None,
    volume_profile: np.ndarray | None,
    config: PolicyConfig,
) -> ExecResult:
    weights = schedule_weights(
        frame=frame,
        side=side,
        policy=policy,
        model=model,
        volume_profile=volume_profile,
        config=config,
    )

    qty = float(qty)
    cum_targets = np.cumsum(weights) * qty
    done = 0.0
    fills: list[tuple[float, float]] = []
    remaining = qty

    for step, (_, row) in enumerate(frame.iterrows()):
        child = max(0.0, float(cum_targets[step] - done))
        if step == len(frame) - 1:
            child = remaining
        child = min(child, remaining)
        px = execution_price(row, qty=child, side=side)
        fills.append((child, px))
        done += child
        remaining -= child

    avg_px = float(sum(q * p for q, p in fills) / qty)
    arrival_mid = float(frame.iloc[0]["mid"])
    end_mid = float(frame.iloc[-1]["mid"])
    sign = 1.0 if side.upper() == "BUY" else -1.0
    is_bps = sign * (avg_px - arrival_mid) / arrival_mid * 1e4

    return ExecResult(
        policy=policy,
        side=side.upper(),
        qty=qty,
        avg_px=avg_px,
        arrival_mid=arrival_mid,
        end_mid=end_mid,
        is_bps=float(is_bps),
    )
