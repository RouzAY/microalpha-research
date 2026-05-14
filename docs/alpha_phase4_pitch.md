# Hedge-fund interview pitch — Phase 4

## One-line pitch

I built a **real LOB alpha-research platform** that starts from single-name execution research, extends to portfolio-style alpha research, and then adds a **cross-asset / cross-venue robustness layer**.

## What phase 4 proves

Phase 4 proves three useful things for a hedge fund:

1. I can build a clean **walk-forward microstructure backtest**.
2. I can compare **classical baselines vs ML vs hybrid methods** in **net bps after costs**.
3. I can reason about **domain shift** and implement a practical safeguard instead of blindly trusting the model.

## Recruiter-friendly interpretation

- **RAMP-R** is the high-performing in-domain hybrid.
- **CAMP-R** is the production-minded extension: use the hybrid when the environment looks familiar, fall back to the safer microstructure policy when it does not.

This is easy to explain to a PM or desk head because it sounds like real risk control, not just academic modeling.

## Best way to present it

"I intentionally kept the system simple enough to audit. The value is not only the score uplift in bps, but the fact that I can show exactly **where ML adds value** and **where the fallback should take over** under transfer risk."
