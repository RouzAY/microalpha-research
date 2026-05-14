from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS


@dataclass
class RidgeSideModel:
    feature_columns: list[str]
    mean_: np.ndarray
    std_: np.ndarray
    coef_buy_: np.ndarray
    coef_sell_: np.ndarray
    horizon: int
    ridge_lambda: float

    def _transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def predict(self, frame: pd.DataFrame, side: str) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=float)
        xs = self._transform(x)
        coef = self.coef_buy_ if side.upper() == "BUY" else self.coef_sell_
        return xs @ coef

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "feature_columns": self.feature_columns,
            "mean_": self.mean_.tolist(),
            "std_": self.std_.tolist(),
            "coef_buy_": self.coef_buy_.tolist(),
            "coef_sell_": self.coef_sell_.tolist(),
            "horizon": self.horizon,
            "ridge_lambda": self.ridge_lambda,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RidgeSideModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_columns=list(payload["feature_columns"]),
            mean_=np.asarray(payload["mean_"], dtype=float),
            std_=np.asarray(payload["std_"], dtype=float),
            coef_buy_=np.asarray(payload["coef_buy_"], dtype=float),
            coef_sell_=np.asarray(payload["coef_sell_"], dtype=float),
            horizon=int(payload["horizon"]),
            ridge_lambda=float(payload["ridge_lambda"]),
        )


def _child_cost(frame: pd.DataFrame, side: str, child_qty: float = 100.0) -> np.ndarray:
    if side.upper() == "BUY":
        size_1 = frame["size_ask_1"].to_numpy(dtype=float)
        px_1 = frame["price_ask_1"].to_numpy(dtype=float)
        px_2 = frame["price_ask_2"].to_numpy(dtype=float)
    else:
        size_1 = frame["size_bid_1"].to_numpy(dtype=float)
        px_1 = frame["price_bid_1"].to_numpy(dtype=float)
        px_2 = frame["price_bid_2"].to_numpy(dtype=float)

    fill_1 = np.minimum(child_qty, size_1)
    rem = child_qty - fill_1
    return (fill_1 * px_1 + rem * px_2) / child_qty


def _future_avg(x: np.ndarray, horizon: int) -> np.ndarray:
    return pd.Series(x).rolling(horizon, min_periods=1).mean().shift(-horizon).to_numpy(dtype=float)


def fit_side_model(day_frames: list[pd.DataFrame], horizon: int = 20, ridge_lambda: float = 100.0) -> RidgeSideModel:
    xs: list[np.ndarray] = []
    ys_buy: list[np.ndarray] = []
    ys_sell: list[np.ndarray] = []

    for frame in day_frames:
        x = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        mid = frame["mid"].to_numpy(dtype=float)

        buy_now = _child_cost(frame, side="BUY", child_qty=100.0)
        sell_now = _child_cost(frame, side="SELL", child_qty=100.0)
        buy_future = _future_avg(buy_now, horizon=horizon)
        sell_future = _future_avg(sell_now, horizon=horizon)

        y_buy = (buy_future - buy_now) / mid
        y_sell = (sell_now - sell_future) / mid
        mask = ~(np.isnan(y_buy) | np.isnan(y_sell))

        xs.append(x[mask])
        ys_buy.append(y_buy[mask])
        ys_sell.append(y_sell[mask])

    x_all = np.vstack(xs)
    y_buy_all = np.concatenate(ys_buy)
    y_sell_all = np.concatenate(ys_sell)

    mean_ = x_all.mean(axis=0)
    std_ = x_all.std(axis=0) + 1e-8
    xs_all = (x_all - mean_) / std_

    eye = np.eye(xs_all.shape[1], dtype=float)
    coef_buy = np.linalg.solve(xs_all.T @ xs_all + ridge_lambda * eye, xs_all.T @ y_buy_all)
    coef_sell = np.linalg.solve(xs_all.T @ xs_all + ridge_lambda * eye, xs_all.T @ y_sell_all)

    return RidgeSideModel(
        feature_columns=list(FEATURE_COLUMNS),
        mean_=mean_,
        std_=std_,
        coef_buy_=coef_buy,
        coef_sell_=coef_sell,
        horizon=horizon,
        ridge_lambda=ridge_lambda,
    )
