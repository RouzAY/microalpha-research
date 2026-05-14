# Interview talk track

## 30-second version

I wanted a project that looked more like real desk research than a Kaggle-style ML exercise. So I built a limit-order-book alpha-research platform around three principles: strong baselines, explicit costs in bps, and honest out-of-sample testing. Then I added a hybrid method, CAMP-R, that combines microstructure signals, a small ML layer, and a fallback under domain shift.

## 90-second version

The project has four layers. Phase 1 is execution research, phase 2 is portfolio-style alpha research, phase 3 is fully real walk-forward testing on AAPL, and phase 4 adds a second venue with Binance to test robustness under transfer.

The central idea is simple: I did not want to rely on ML alone. I used a microstructure-first signal, then let a small learned layer help where the data supported it, and finally added a gate so that when the warmup distribution looks too different from the training universe, the system falls back to the safer microstructure logic.

That produces a result that is easier to explain to a PM: in-domain you keep the upside of the hybrid, and out-of-domain you avoid being overconfident.

## Key numbers to mention

- Combined benchmark: **CAMP-R = 0.1711 mean net bps/step**.
- Total net bps: **29194.6**.
- Positive episodes: **9/10**.
- Episode t-stat: **3.70**.
- In-domain AAPL: **0.3148**.
- External Binance: **0.0309**.

## Questions you should be ready for

### Why is this interesting beyond backtest cosmetics?
Because the project explicitly separates gross edge, costs, turnover, and robustness. It is not just a signal score.

### Why not use a heavier deep model?
Because for a recruiting artifact, auditability matters. I wanted a method that is easy to reason about and easy to benchmark against classical alternatives.

### What are the limitations?
The benchmark is real but still compact. It is a proof of research quality and experimental discipline, not a claim of production-scale live alpha.

### What would you do next?
Scale the same framework to a wider real universe, add portfolio construction and richer risk controls, and test whether the fallback logic remains useful under broader domain shift.
