from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .universe import FAST_HORIZONS, HORIZONS, SyntheticEpisode


@dataclass
class RidgeEnsemble:
    coef_by_horizon: dict[int, np.ndarray]
    regime_reliability: dict[int, dict[int, float]]

    def predict_horizon(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return x @ self.coef_by_horizon[horizon]


def fit_ridge_ensemble(episodes: list[SyntheticEpisode], ridge_lambda: float = 45.0) -> dict[int, np.ndarray]:
    x = np.vstack([episode.features[symbol] for episode in episodes for symbol in episode.symbols])
    eye = np.eye(x.shape[1], dtype=float)

    coefs: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        y = np.concatenate(
            [episode.horizon_returns[symbol][horizon] for episode in episodes for symbol in episode.symbols]
        )
        coefs[horizon] = np.linalg.solve(x.T @ x + ridge_lambda * eye, x.T @ y)
    return coefs


def estimate_regime_reliability(
    episodes: list[SyntheticEpisode],
    coef_by_horizon: dict[int, np.ndarray],
) -> dict[int, dict[int, float]]:
    reliability: dict[int, dict[int, float]] = {regime: {} for regime in range(4)}

    for regime in range(4):
        for horizon in HORIZONS:
            preds: list[float] = []
            targets: list[float] = []
            coef = coef_by_horizon[horizon]
            for episode in episodes:
                for symbol in episode.symbols:
                    mask = episode.regimes[symbol] == regime
                    if not np.any(mask):
                        continue
                    preds.extend((episode.features[symbol][mask] @ coef).tolist())
                    targets.extend(episode.realized_returns[symbol][mask].tolist())
            if len(preds) < 5:
                reliability[regime][horizon] = 0.0
                continue
            corr = np.corrcoef(preds, targets)[0, 1]
            reliability[regime][horizon] = max(float(corr), 0.0)
    return reliability


def fit_full_model(episodes_train: list[SyntheticEpisode], episodes_val: list[SyntheticEpisode]) -> RidgeEnsemble:
    coefs = fit_ridge_ensemble(episodes_train)
    reliability = estimate_regime_reliability(episodes_val, coefs)
    return RidgeEnsemble(coef_by_horizon=coefs, regime_reliability=reliability)
