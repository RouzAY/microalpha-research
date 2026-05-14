# Phase 3 fully-real walk-forward alpha benchmark

## What changes in phase 3

Phase 3 removes the synthetic multi-symbol universe and evaluates everything on **real AAPL limit-order-book days only**.
The benchmark is now a strict **anchored walk-forward** setup: each evaluation day is traded using only previous days plus a same-day warmup window.

## Method

Our phase-3 method is **RAMP-R** = **Regime-Adaptive Microstructure Portfolio on Real data**.

RAMP-R combines:
1. a microstructure signal based on microprice displacement / imbalance;
2. a ridge model trained on all prior real days;
3. an online morning regime test to decide whether the classic imbalance signal should keep or flip sign;
4. sparse trading with a fixed holding period;
5. an effective execution-cost model proportional to spread × turnover.

## Evaluation window

- Symbol: AAPL
- Real days available: 10
- Walk-forward evaluation days: 2025-06-03, 2025-06-04, 2025-06-05, 2025-06-06, 2025-06-07, 2025-06-08, 2025-06-09, 2025-06-10, 2025-06-11
- Final held-out days: 2025-06-10, 2025-06-11
- Horizon / hold: 30 / 30
- Warmup rows: 800
- Threshold quantile: 0.85
- Effective cost coefficient: 0.12

## Full walk-forward summary

|   mean_net_bps |   total_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |   hit_rate |   n_days |   positive_days |   tstat_daily_total | strategy      |
|---------------:|----------------:|-----------------:|----------------:|-----------:|---------------:|-----------:|---------:|----------------:|--------------------:|:--------------|
|         0.3148 |      26528.2178 |           0.3298 |          0.0150 |     0.0078 |         0.1559 |     0.0813 |        9 |               8 |              3.3435 | ramp_real     |
|         0.2582 |      21753.9402 |           0.2694 |          0.0113 |     0.0094 |         0.1655 |     0.0863 |        9 |               7 |              2.2540 | online_regime |
|         0.1446 |      12182.2764 |           0.1591 |          0.0145 |     0.0070 |         0.1390 |     0.0712 |        9 |               6 |              0.9791 | ridge_static  |
|        -0.0284 |      -2396.0139 |          -0.0172 |          0.0113 |     0.0094 |         0.1655 |     0.0833 |        9 |               5 |             -0.1902 | micro_fixed   |
|        -0.1140 |      -9606.8110 |          -0.1011 |          0.0129 |     0.0067 |         0.1153 |     0.0553 |        9 |               5 |             -0.5959 | momentum      |
|        -0.1287 |     -10848.1120 |          -0.1173 |          0.0115 |     0.0080 |         0.1328 |     0.0658 |        9 |               2 |             -2.1099 | signed_flow   |

## Final held-out summary

|   mean_net_bps |   total_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |   hit_rate |   n_days |   positive_days |   tstat_daily_total | strategy      |
|---------------:|----------------:|-----------------:|----------------:|-----------:|---------------:|-----------:|---------:|----------------:|--------------------:|:--------------|
|         0.6301 |      11785.8967 |           0.6455 |          0.0155 |     0.0118 |         0.2069 |     0.1085 |        2 |               2 |              2.2463 | online_regime |
|         0.3971 |       7428.4613 |           0.4175 |          0.0204 |     0.0098 |         0.1989 |     0.1062 |        2 |               2 |              1.4165 | ramp_real     |
|        -0.1680 |      -3142.2885 |          -0.1585 |          0.0095 |     0.0061 |         0.0962 |     0.0436 |        2 |               0 |             -3.3951 | signed_flow   |
|        -0.1951 |      -3650.4217 |          -0.1795 |          0.0156 |     0.0074 |         0.1219 |     0.0494 |        2 |               1 |             -0.3251 | momentum      |
|        -0.4200 |      -7857.4176 |          -0.4012 |          0.0189 |     0.0095 |         0.1860 |     0.0892 |        2 |               0 |             -1.0278 | ridge_static  |
|        -0.6610 |     -12364.0574 |          -0.6455 |          0.0155 |     0.0118 |         0.2069 |     0.0948 |        2 |               0 |             -2.4192 | micro_fixed   |

## Daily totals

| strategy      | day        |   total_net_bps |   mean_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |   hit_rate |
|:--------------|:-----------|----------------:|---------------:|-----------------:|----------------:|-----------:|---------------:|-----------:|
| micro_fixed   | 2025-06-03 |         2858.89 |           0.30 |             0.30 |            0.01 |       0.01 |           0.10 |       0.05 |
| momentum      | 2025-06-03 |        -4925.83 |          -0.51 |            -0.50 |            0.01 |       0.01 |           0.09 |       0.04 |
| online_regime | 2025-06-03 |         2858.89 |           0.30 |             0.30 |            0.01 |       0.01 |           0.10 |       0.05 |
| ramp_real     | 2025-06-03 |         4397.53 |           0.46 |             0.47 |            0.01 |       0.00 |           0.09 |       0.05 |
| ridge_static  | 2025-06-03 |         5161.44 |           0.54 |             0.55 |            0.01 |       0.00 |           0.09 |       0.05 |
| signed_flow   | 2025-06-03 |        -1975.93 |          -0.21 |            -0.20 |            0.01 |       0.01 |           0.11 |       0.06 |
| micro_fixed   | 2025-06-04 |        -2227.27 |          -0.22 |            -0.21 |            0.01 |       0.01 |           0.12 |       0.06 |
| momentum      | 2025-06-04 |          276.40 |           0.03 |             0.03 |            0.00 |       0.00 |           0.04 |       0.02 |
| online_regime | 2025-06-04 |        -2227.27 |          -0.22 |            -0.21 |            0.01 |       0.01 |           0.12 |       0.06 |
| ramp_real     | 2025-06-04 |         -106.95 |          -0.01 |            -0.00 |            0.01 |       0.00 |           0.08 |       0.04 |
| ridge_static  | 2025-06-04 |          101.90 |           0.01 |             0.02 |            0.01 |       0.00 |           0.07 |       0.04 |
| signed_flow   | 2025-06-04 |        -2751.46 |          -0.27 |            -0.26 |            0.01 |       0.01 |           0.13 |       0.06 |
| micro_fixed   | 2025-06-05 |         1301.10 |           0.15 |             0.15 |            0.01 |       0.01 |           0.12 |       0.06 |
| momentum      | 2025-06-05 |        -6995.30 |          -0.78 |            -0.77 |            0.01 |       0.01 |           0.09 |       0.04 |
| online_regime | 2025-06-05 |         1301.10 |           0.15 |             0.15 |            0.01 |       0.01 |           0.12 |       0.06 |
| ramp_real     | 2025-06-05 |         5261.59 |           0.59 |             0.60 |            0.01 |       0.00 |           0.05 |       0.03 |
| ridge_static  | 2025-06-05 |         5252.18 |           0.59 |             0.59 |            0.01 |       0.00 |           0.05 |       0.03 |
| signed_flow   | 2025-06-05 |        -3053.72 |          -0.34 |            -0.33 |            0.01 |       0.01 |           0.13 |       0.07 |
| micro_fixed   | 2025-06-06 |         -203.97 |          -0.02 |            -0.01 |            0.02 |       0.01 |           0.25 |       0.13 |
| momentum      | 2025-06-06 |         6886.33 |           0.75 |             0.78 |            0.02 |       0.01 |           0.29 |       0.15 |
| online_regime | 2025-06-06 |         -203.97 |          -0.02 |            -0.01 |            0.02 |       0.01 |           0.25 |       0.13 |
| ramp_real     | 2025-06-06 |         1046.25 |           0.11 |             0.14 |            0.02 |       0.01 |           0.29 |       0.15 |
| ridge_static  | 2025-06-06 |         2310.56 |           0.25 |             0.27 |            0.02 |       0.01 |           0.26 |       0.13 |
| signed_flow   | 2025-06-06 |         2345.12 |           0.26 |             0.27 |            0.01 |       0.01 |           0.17 |       0.09 |
| micro_fixed   | 2025-06-07 |         1805.82 |           0.19 |             0.21 |            0.02 |       0.01 |           0.19 |       0.09 |
| momentum      | 2025-06-07 |        -5607.93 |          -0.58 |            -0.57 |            0.02 |       0.01 |           0.15 |       0.07 |
| online_regime | 2025-06-07 |         1805.82 |           0.19 |             0.21 |            0.02 |       0.01 |           0.19 |       0.09 |
| ramp_real     | 2025-06-07 |          571.12 |           0.06 |             0.08 |            0.02 |       0.01 |           0.25 |       0.12 |
| ridge_static  | 2025-06-07 |         2012.87 |           0.21 |             0.23 |            0.02 |       0.01 |           0.23 |       0.12 |
| signed_flow   | 2025-06-07 |        -1622.88 |          -0.17 |            -0.15 |            0.02 |       0.01 |           0.19 |       0.09 |
| micro_fixed   | 2025-06-08 |         5799.35 |           0.67 |             0.68 |            0.01 |       0.01 |           0.18 |       0.11 |
| momentum      | 2025-06-08 |         4407.54 |           0.51 |             0.52 |            0.01 |       0.00 |           0.05 |       0.03 |
| online_regime | 2025-06-08 |         5799.35 |           0.67 |             0.68 |            0.01 |       0.01 |           0.18 |       0.11 |
| ramp_real     | 2025-06-08 |         1424.53 |           0.17 |             0.17 |            0.01 |       0.00 |           0.09 |       0.04 |
| ridge_static  | 2025-06-08 |         -302.08 |          -0.04 |            -0.02 |            0.01 |       0.00 |           0.08 |       0.04 |
| signed_flow   | 2025-06-08 |        -1339.76 |          -0.16 |            -0.14 |            0.01 |       0.01 |           0.12 |       0.06 |
| micro_fixed   | 2025-06-09 |          634.13 |           0.07 |             0.07 |            0.01 |       0.01 |           0.12 |       0.07 |
| momentum      | 2025-06-09 |            2.39 |           0.00 |             0.01 |            0.01 |       0.01 |           0.09 |       0.05 |
| online_regime | 2025-06-09 |          634.13 |           0.07 |             0.07 |            0.01 |       0.01 |           0.12 |       0.07 |
| ramp_real     | 2025-06-09 |         6505.68 |           0.68 |             0.69 |            0.01 |       0.01 |           0.16 |       0.08 |
| ridge_static  | 2025-06-09 |         5502.82 |           0.57 |             0.58 |            0.01 |       0.01 |           0.11 |       0.05 |
| signed_flow   | 2025-06-09 |          692.81 |           0.07 |             0.08 |            0.01 |       0.01 |           0.15 |       0.07 |
| micro_fixed   | 2025-06-10 |        -8737.41 |          -0.95 |            -0.94 |            0.01 |       0.01 |           0.22 |       0.10 |
| momentum      | 2025-06-10 |         3788.84 |           0.41 |             0.42 |            0.01 |       0.01 |           0.10 |       0.05 |
| online_regime | 2025-06-10 |         8516.32 |           0.92 |             0.94 |            0.01 |       0.01 |           0.22 |       0.12 |
| ramp_real     | 2025-06-10 |         1092.03 |           0.12 |             0.14 |            0.02 |       0.01 |           0.25 |       0.13 |
| ridge_static  | 2025-06-10 |         -106.25 |          -0.01 |             0.01 |            0.02 |       0.01 |           0.26 |       0.12 |
| signed_flow   | 2025-06-10 |        -1108.37 |          -0.12 |            -0.11 |            0.01 |       0.01 |           0.11 |       0.06 |
| micro_fixed   | 2025-06-11 |        -3626.65 |          -0.38 |            -0.36 |            0.02 |       0.01 |           0.19 |       0.09 |
| momentum      | 2025-06-11 |        -7439.26 |          -0.78 |            -0.76 |            0.02 |       0.01 |           0.15 |       0.05 |
| online_regime | 2025-06-11 |         3269.58 |           0.34 |             0.36 |            0.02 |       0.01 |           0.19 |       0.10 |
| ramp_real     | 2025-06-11 |         6336.43 |           0.67 |             0.69 |            0.02 |       0.01 |           0.15 |       0.08 |
| ridge_static  | 2025-06-11 |        -7751.17 |          -0.82 |            -0.80 |            0.02 |       0.01 |           0.11 |       0.06 |
| signed_flow   | 2025-06-11 |        -2033.92 |          -0.21 |            -0.20 |            0.01 |       0.00 |           0.08 |       0.03 |

## Headline

Across all real out-of-sample walk-forward days, **ramp_real** finishes first with **0.3148 mean net bps/step**, **26528.2 total net bps**, and **8/9 positive days**.

On the strict final held-out block, **online_regime** is first at **0.6301 mean net bps/step**.

## Honesty note

This phase is **fully real in the sense of data and walk-forward evaluation**, but it is still limited by the bundled sample: only one symbol and only ten days are available.
So this is a strong *research-platform* result, not a claim of production-ready, multi-asset live alpha.
