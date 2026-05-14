# MicroAlpha Execution + Alpha Research Lab

A research platform for **limit-order-book execution**, **short-horizon alpha evaluation**, and **robustness under market shift**, with explicit transaction-cost modeling and results reported in **net basis points after costs**.

This repository studies three related but distinct questions:

1. Can order-book information improve **execution quality** relative to standard schedules such as **TWAP** and **VWAP**?
2. Can microstructure-aware signals remain **net positive after costs** on real historical data?
3. Can a strategy remain useful when the target market differs from the training market?

The project is organized as a four-phase benchmark:

1. **Phase 1 — Execution research**  
   Event-driven benchmark on real and synthetic LOB data with implementation-shortfall metrics in basis points.

2. **Phase 2 — Alpha research on a real-calibrated synthetic universe**  
   Cost-aware market-neutral benchmark with multiple cross-sectional baselines.

3. **Phase 3 — Fully real walk-forward alpha benchmark**  
   Anchored day-by-day evaluation using only real AAPL LOB days.

4. **Phase 4 — Real multi-asset / cross-venue robustness**  
   AAPL walk-forward plus Binance transfer test with a domain-shift fallback mechanism.

The goal is not to claim production live trading.  
The goal is to make the contribution of microstructure-aware methods **legible under costs, turnover, and regime shift**.

---

## Why this repository is useful

Many trading side projects focus on prediction alone, rely on toy backtests, or omit realistic cost accounting.

This repository instead emphasizes:

- real microstructure data
- walk-forward evaluation
- spread-based transaction costs
- clean, comparable baselines
- net bps, turnover, hit rate, and episode-level stability
- hybrid ML + microstructure decision rules
- robustness under transfer shift

---

## Key results

### Phase 4 — combined real multi-asset benchmark

On the combined benchmark  
(**AAPL / XNAS** walk-forward + **BNBUSDT / BINANCE** external transfer test):

- **CAMP-R**: **0.1711 mean net bps / step**
- **RAMP-R**: 0.1643
- **online_regime**: 0.1431
- **ridge_static**: 0.0798

Headline summary:

- **29,194.6 total net bps**
- **9 positive episodes out of 10**
- **episode t-stat ≈ 3.70**

### Phase 3 — fully real walk-forward benchmark

On the default fully real anchored walk-forward evaluation:

- **RAMP-R**: **0.3148 mean net bps / step**
- **online_regime**: 0.2582
- **ridge_static**: 0.1446
- **micro_fixed**: -0.0284
- **momentum**: -0.1140
- **signed_flow**: -0.1287

RAMP-R also delivers:

- **26,528.2 total net bps**
- **7 positive days out of 9**

### Phase 1 — execution benchmark

Phase 1 compares:

- **TWAP**
- **VWAP**
- **Liquidity-only**
- **Alpha-only**
- **MALP**

using **implementation shortfall in bps** as the main execution metric.

---

## Project structure

The repository is best read as a progression:

**execution** → **controlled alpha sandbox** → **real historical alpha** → **robustness under shift**

This separation is deliberate.

- **Phase 1** answers an execution question: if a parent order is already known, how should it be split into child orders?
- **Phases 3 and 4** answer alpha questions: should the strategy take directional exposure, and how should it control model trust under drift?
- **Phase 2** provides an intermediate sandbox between execution and fully real alpha evaluation.

Because these are different problems, they use different baselines.

- In **Phase 1**, the relevant baselines are **TWAP** and **VWAP**, because the problem is execution.
- In **Phases 3 and 4**, the relevant baselines are directional rules such as **momentum**, **signed flow**, **micro_fixed**, **ridge_static**, and **online_regime**.

---

## Main methods

### Phase 1 — Execution research

A clean event-driven execution benchmark on real and synthetic LOB data.

Included policies:

- **TWAP**
- **VWAP**
- **Liquidity-only**
- **Alpha-only**
- **MALP** = *Microstructure Alpha + Liquidity Policy*

#### Why it matters

This phase answers the execution question:

> can a microstructure-aware policy reduce implementation shortfall relative to naive schedules?

MALP combines two ingredients:

- **local liquidity quality**, which tells us where the book looks easier to access;
- **execution alpha / urgency**, which tells us when waiting may become more expensive.

This phase establishes that order-book information is useful even before any directional trading problem is introduced.

---

### Phase 2 — Alpha research on a real-calibrated synthetic universe

Phase 2 adds:

- a **synthetic multi-symbol universe** calibrated from real AAPL order-book snapshots;
- **cross-sectional market-neutral backtests** with explicit turnover costs;
- **RAMP** = *Regime-Adaptive Multi-horizon Portfolio*.

Baselines:

- **uniform**
- **imbalance**
- **momentum**
- **ridge5**
- **RAMP** (ours)

#### Why it matters

This phase bridges pure execution research and alpha research by asking:

> can short-horizon microstructure structure be turned into cost-aware cross-sectional alpha?

Phase 2 is intentionally not the strongest evidence in the project.  
Its role is to provide a controlled research sandbox in which multi-horizon prediction, regime adaptation, and cost-aware allocation can be tested before moving to fully real historical evaluation.

---

### Phase 3 — Fully real walk-forward alpha benchmark

Phase 3 removes the synthetic test universe and works only with real AAPL LOB days bundled in the repository.

It adds:

- **anchored walk-forward evaluation**
- a **same-day warmup regime detector**
- a **ridge model fit only on prior real days**
- sparse position-taking with explicit spread-based cost accounting
- **RAMP-R** = *Regime-Adaptive Microstructure Portfolio on Real data*

Baselines:

- **momentum**
- **signed_flow**
- **micro_fixed**
- **ridge_static**
- **online_regime**
- **RAMP-R** (ours)

#### RAMP-R in one paragraph

RAMP-R combines three ingredients:

1. a microstructure signal based on **microprice displacement / imbalance**;
2. a **ridge predictor** trained only on prior real days;
3. an **online regime switch** estimated from the first part of the current day.

If the warmup segment suggests that the usual imbalance relationship is informative enough, the model trusts the current-day sign.
Otherwise, it falls back to a more stable prior-day logic.
The final score is thresholded into sparse positions, held for a fixed horizon, and evaluated after spread-based turnover costs.

#### Why it matters

This is the first phase that answers the central alpha-research question:

> can the strategy remain net positive on fully real out-of-sample days once costs are accounted for?

---

### Phase 4 — Real multi-asset / cross-venue robustness benchmark

Phase 4 keeps the fully real AAPL walk-forward benchmark and adds a second real market:

- **AAPL / XNAS** historical LOB walk-forward
- **BNBUSDT / BINANCE** as an external transfer test

It adds:

- a real **cross-asset / cross-venue** net-bps benchmark
- **CAMP-R** = *Cross-Asset Adaptive Microstructure Portfolio on Real data*
- a simple **domain-shift gate** deciding when to trust the hybrid ML layer and when to fall back to safer microstructure logic

#### Why it matters

This phase tests whether the signal logic is merely in-domain fit, or whether it can degrade gracefully under transfer.

CAMP-R does not simply ask whether one model is strongest on average.  
It asks a more realistic question:

> when the current market looks different from the training market, should the same model be trusted to the same extent?

That is the core robustness contribution of the project.

---

## Quickstart

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run the four main benchmarks:

```bash
python scripts/run_benchmark.py
python scripts/run_alpha_research_phase2.py
python scripts/run_alpha_research_phase3.py
python scripts/run_alpha_research_phase4.py
```

Launch the dashboard:

```bash
streamlit run app/demo_dashboard.py
```

---

## Repository layout

```text
.
├── app/
│   ├── demo_dashboard.py
│   ├── common.py
│   └── pages/
├── data/
│   ├── optiver/
│   └── binance/
├── docs/
│   ├── method_note.md
│   ├── alpha_research_method.md
│   ├── alpha_phase3_method.md
│   └── alpha_phase4_method.md
├── reports/
│   ├── benchmark_summary.md
│   ├── alpha_phase2_summary.md
│   ├── alpha_phase3_summary.md
│   ├── alpha_phase4_summary.md
│   ├── *.png
│   ├── *.csv
│   └── *.json
├── scripts/
│   ├── run_benchmark.py
│   ├── run_alpha_research_phase2.py
│   ├── run_alpha_research_phase3.py
│   ├── run_alpha_research_phase4.py
│   └── build_phase4_episode_summary.py
├── src/
│   └── microalpha_exec_lab/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Dashboard

The Streamlit dashboard is organized to mirror the project logic.

Suggested reading order:

1. **Overview**
2. **Phase 1 — Execution**
3. **Phase 3 — Real Alpha**
4. **Phase 4 — Robustness**
5. **Methods & Limitations**
6. **Phase 2 — Appendix**

The dashboard is designed to clarify:

- what each phase studies
- which baselines are relevant in each phase
- how the reported metrics should be interpreted
- what the project models explicitly and what it does not

---

## Scope and limitations

This repository explicitly models:

- spread and displayed top-of-book liquidity
- level-1 / level-2 imbalance
- signed trade flow
- spread-based transaction costs
- walk-forward retraining
- warmup adaptation
- domain-shift-aware fallback logic

It does **not** explicitly model:

- exact queue position
- passive fill probability
- detailed queue cancellation dynamics
- latency-sensitive exchange mechanics
- full production-grade order management

The correct interpretation is therefore:

> **a research-grade microstructure benchmark and alpha-validation platform, not a full exchange simulator**

---

## Why the benchmark design is coherent

The benchmark is designed so that each phase answers one main question with the appropriate comparators.

- **Phase 1** compares execution policies against **TWAP** and **VWAP**, because the task is parent-order scheduling.
- **Phase 2** tests cross-sectional alpha construction in a controlled sandbox.
- **Phase 3** evaluates directional alpha on fully real historical data in a strict walk-forward setting.
- **Phase 4** studies robustness under cross-market shift and adaptive model trust.

This separation is important because it prevents execution, prediction, and robustness from being mixed into a single uninterpretable benchmark.

---

## Core quantities used throughout the project

The repository relies on standard market-microstructure quantities, including:

- **midprice**
- **spread**
- **level-1 / level-2 imbalance**
- **microprice**
- **signed trade flow**
- **rolling returns**
- **turnover**
- **net bps after costs**

In the execution phase, the main metric is **implementation shortfall in basis points**.  
In the alpha phases, the main metric is **mean net bps / step**, together with cumulative net bps, turnover, hit rate, positive episodes, and episode-level stability.

---

## Reproducibility

The repository is organized so that the main results can be regenerated from the benchmark scripts and inspected through the dashboard.

The intended workflow is:

1. run the benchmark scripts,
2. generate summary files and figures in `reports/`,
3. inspect phase-by-phase behavior in the dashboard.

If you modify the data splits, cost settings, or feature definitions, results will change accordingly.

---

## Suggested entry points

If you only inspect a few files first, start with:

- `app/demo_dashboard.py`
- `app/pages/1_Phase_1_Execution.py`
- `app/pages/2_Phase_3_Real_Alpha.py`
- `app/pages/3_Phase_4_Robustness.py`
- `docs/method_note.md`

If you want to run only one benchmark first, start with:

```bash
python scripts/run_alpha_research_phase4.py
```

and then open:

```bash
streamlit run app/demo_dashboard.py
```

---

## Summary

The central contribution of this repository is not a claim of perfect market simulation.

The contribution is a structured research framework that separates:

- **execution quality**
- **directional alpha**
- **robustness under drift and transfer shift**

In that sense, the project is best read as:

**execute better** → **experiment safely** → **validate on real data** → **control model risk under shift**
