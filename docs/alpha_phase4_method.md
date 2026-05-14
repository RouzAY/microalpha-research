# Phase 4 method note — CAMP-R

## Goal

Phase 4 is designed for an **alpha-research recruiting story**, not only an execution story.
The question is no longer just *"can we trade one stock well?"*.
It becomes:

> Can a microstructure strategy remain profitable when the target episode comes from a **different asset and a different venue**?

## Benchmark structure

The benchmark now contains two real tracks:

- **AAPL / XNAS**: anchored walk-forward, day by day.
- **BNBUSDT / BINANCE**: external real historical test.

This makes phase 4 useful for interviews because it demonstrates:

- in-domain alpha research;
- out-of-domain robustness;
- a simple but credible *model-risk control* mechanism.

## CAMP-R

**CAMP-R** stands for **Cross-Asset Adaptive Microstructure Portfolio on Real data**.

It is intentionally simple:

1. fit the same ridge-based transfer model as before on the training universe;
2. compute a warmup-domain-shift score from the target episode;
3. if the target episode looks familiar, use the stronger hybrid score (**RAMP-R mode**);
4. if the target episode looks too different, fall back to the safer microstructure-only regime logic.

## Why this is defensible

Many projects fail in interviews because they only show one of two extremes:

- either a fancy model with no robustness story;
- or a robust baseline with no ML contribution.

CAMP-R gives a clear answer to both:

- **ML helps** on the in-domain AAPL walk-forward problem;
- **fallback logic protects** performance when the new venue looks far from training.

## Practical pitch

The pitch is not *"I solved live multi-asset trading"*.
The pitch is:

> I built a real-data research platform showing when the ML layer helps, when it hurts, and how to switch safely under domain shift.
