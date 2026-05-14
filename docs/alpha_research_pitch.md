# Phase 2 recruiting pitch — alpha researcher version

## One-line CV bullet

Built a regime-aware multi-horizon alpha research platform calibrated from real limit-order-book features, with market-neutral portfolio backtests under explicit turnover costs, and a custom method (RAMP) that outperformed simpler baselines in net bps.

## Slightly stronger CV bullet

Designed and benchmarked a cost-aware cross-sectional alpha pipeline on a synthetic multi-symbol universe calibrated from real LOB snapshots; proposed a Regime-Adaptive Multi-horizon Portfolio (RAMP) that improved net bps and reduced turnover versus single-horizon and uniform-ensemble baselines.

## 30-second interview version

I started from a microstructure execution project and extended it into an alpha-research benchmark. I calibrated a realistic feature pool from real order-book data, generated a held-out multi-symbol universe, trained multiple horizon-specific alpha models, and then built a regime-aware ensemble with explicit cost shrinkage and market-neutral portfolio construction. The point was not to claim live alpha, but to show a full research workflow with out-of-sample comparison in basis points.

## 60-second interview version

A lot of candidate projects stop at signal prediction. I wanted something closer to how a hedge fund would evaluate research, so I added three layers: a synthetic but realistic multi-symbol universe calibrated from real LOB features, an explicit cost-aware portfolio wrapper, and a proper baseline suite. My custom method, RAMP, combines fast and slow horizon predictors using validation-estimated regime reliability and observable microstructure gating. On the held-out benchmark, it produced the best net bps while also reducing turnover materially versus the strongest simple baselines. So the contribution is both predictive and implementation-aware.

## What makes the project convincing

- It has a **clear benchmark**.
- It uses **out-of-sample validation and testing**.
- It reports **net bps**, not just prediction accuracy.
- It compares against **stronger baselines than a toy project**.
- It has an explicit **portfolio construction** step.
- It is honest about what is **real** and what is **synthetic**.

## What to say if they ask “is this live alpha?”

No. The feature calibration source is real, but the multi-symbol portfolio testbed is synthetic. The value of the project is that it demonstrates the full alpha-research loop: feature engineering, target design, model training, validation, cost-aware portfolio construction, and benchmark discipline. The next step in a real fund setting would be to swap the synthetic universe for internal historical data and rerun exactly the same framework.

## Good closing line

The project is meant to show that I can move from microstructure intuition to a complete alpha-research process and defend results in portfolio-level, post-cost terms.
