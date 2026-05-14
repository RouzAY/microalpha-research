# Phase 3 method note — RAMP-R

## Objective

Phase 3 moves from a real-calibrated synthetic universe to a **fully real walk-forward benchmark**.
The objective is not to claim production alpha, but to answer a defensible research question:

> given only prior real days and a short same-day warmup window, can a regime-aware microstructure model stay net positive after explicit execution costs?

## Data protocol

- instrument: bundled real **AAPL** LOB snapshots and trades;
- horizon: `H = 30` rows;
- same-day warmup: first `800` rows;
- anchored walk-forward: every evaluation day uses only **previous days** for training;
- final held-out block: last 2 days.

## Features

The signal stack uses spread, L1/L2 imbalance, microprice displacement, rolling signed trade flow, short-horizon returns, volume/trade-count proxies, and interaction terms such as `spread_bps × imbalance`.

## Baselines

- **momentum**: sign of recent return;
- **signed_flow**: sign of rolling signed order flow;
- **micro_fixed**: classic microstructure signal with a fixed sign from prior days;
- **ridge_static**: prior-day ridge predictor without same-day regime adaptation;
- **online_regime**: microstructure signal plus same-day sign adaptation.

## Our method: RAMP-R

RAMP-R = **Regime-Adaptive Microstructure Portfolio on Real data**.

It has five steps:

1. **Prior-day model fit**: fit a ridge model on all previous real days to predict future mid-price return over horizon `H`.
2. **Microstructure anchor**: compute the normalized microprice-displacement signal.
3. **Same-day regime test**: during the warmup window estimate whether the sign of the microstructure relationship is currently positive or negative.
4. **Hybrid score**: combine the microstructure signal with the standardized ridge prediction.
5. **Sparse execution-aware trading**: threshold the signal, hold it for a fixed horizon, and subtract a spread-based turnover cost.

## Why this is stronger than a static signal

A static imbalance rule implicitly assumes stationarity.
The real days in the sample do not behave that cleanly: some days preserve the classic sign, while others flip.

RAMP-R is stronger because it explicitly separates:
- **slow information** from previous days,
- **fast regime information** from the current morning,
- **execution selectivity** via thresholding and holding.
