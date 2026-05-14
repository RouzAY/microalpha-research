# Phase 3 recruiting pitch

## One-line version

I built a **fully real walk-forward microstructure alpha benchmark** and designed a hybrid model, **RAMP-R**, that combines prior-day machine learning with same-day regime adaptation and beats static baselines in net bps.

## Longer version

Most toy LOB projects stop at either a simulator, a one-shot signal plot, or a synthetic backtest.
I wanted something more credible for hedge-fund recruiting.
So I built a three-phase research platform:

1. **execution research** with implementation-shortfall metrics in bps;
2. **alpha research** on a cost-aware real-calibrated synthetic universe;
3. **fully real walk-forward alpha evaluation** on bundled order-book days.

The phase-3 method, **RAMP-R**, does three things at once:
- uses a ridge model trained only on previous real days,
- detects whether the microstructure regime has flipped using the current morning,
- trades sparsely enough to stay meaningful after explicit spread costs.

## Honest framing

Do not oversell it as live production alpha.
Say instead:

> It is a serious, reproducible research artifact built on real LOB data, but the sample is still small and single-name. The point was to demonstrate the entire research loop and show that the method survives an out-of-sample, fully real, cost-aware protocol.
