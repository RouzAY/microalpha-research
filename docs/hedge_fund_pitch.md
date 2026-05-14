# Hedge-fund pitch notes

## One-line pitch

Built a reproducible market-microstructure execution research project on real and synthetic LOB data, and designed a trainable policy that reduced held-out implementation shortfall from roughly 4.9 bps to 1.9 bps on the included benchmark.

## CV bullets

- Built an event-driven execution benchmark on real AAPL limit-order-book data with synthetic stress testing and basis-point level evaluation.
- Implemented standard execution baselines (TWAP, VWAP) and adaptive heuristics, then proposed **MALP**, a trainable microstructure-alpha + liquidity policy.
- Achieved out-of-sample improvement of about **3 bps vs TWAP** on the held-out benchmark using only interpretable book/trade features and a lightweight ridge model.
- Packaged the project as a clean, reproducible research repo with one-command benchmarking and report generation.

## What recruiters usually like here

- you used **real market data**, not only toy simulation;
- you report **bps**, not vague accuracy metrics;
- you separate **train / validation / test**;
- you compare against recognized baselines;
- your method is simple enough to audit and extend.

## Honest framing

Do not say this proves a production trading edge.
Say instead:

- this proves I can formulate a market microstructure problem rigorously;
- I can build a reproducible benchmark;
- I can propose a new method and evaluate it out-of-sample;
- I understand how to reason in bps, costs, execution quality, and data leakage.

## Best follow-up project after this one

To get closer to a pure alpha researcher profile, extend this repo with:

1. short-horizon return prediction across many symbols;
2. cross-sectional ranking and capacity analysis;
3. portfolio construction under realistic costs;
4. robustness by regime, day, volatility bucket, and market state.
