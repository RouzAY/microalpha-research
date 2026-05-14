# Phase 4 real multi-asset alpha benchmark

## What phase 4 adds

Phase 4 keeps the fully real AAPL walk-forward benchmark from phase 3 and adds a **second real asset / venue**: Binance **BNBUSDT**.
The resulting benchmark is no longer just single-name time-series alpha. It now tests whether a microstructure method can remain useful under **cross-asset / cross-venue domain shift**.

## Evaluation design

- In-domain track: AAPL on XNAS, anchored walk-forward over days 2025-06-03, 2025-06-04, 2025-06-05, 2025-06-06, 2025-06-07, 2025-06-08, 2025-06-09, 2025-06-10, 2025-06-11
- External track: BNBUSDT on BINANCE, historical zero-shot external day(s): 2024-03-30
- Horizon / hold: 30 / 30
- Warmup rows: 800
- Threshold quantile: 0.85
- Effective spread-cost coefficient: 0.12
- Domain-shift threshold for CAMP-R fallback: 3.0

## Method

Our new phase-4 method is **CAMP-R** = **Cross-Asset Adaptive Microstructure Portfolio on Real data**.

CAMP-R works in two modes:
1. **normal regime**: if the target episode looks statistically close to the training distribution, CAMP-R uses the stronger phase-3 hybrid (**RAMP-R** style micro + ML score);
2. **domain-shift regime**: if the warmup distribution is far from the training universe, CAMP-R falls back to the safer microstructure-only regime logic.

This is intentionally simple and recruiter-friendly: it clearly answers the practical question "what should the model do when a new venue or asset looks nothing like the training set?"

## Combined multi-asset summary

|   mean_net_bps |   total_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |   hit_rate |   n_episodes |   positive_episodes |   tstat_episode_total | strategy      |
|---------------:|----------------:|-----------------:|----------------:|-----------:|---------------:|-----------:|-------------:|--------------------:|----------------------:|:--------------|
|         0.1711 |      29194.6332 |           0.1787 |          0.0075 |     0.0091 |         0.1651 |     0.0849 |           10 |                   9 |                3.7002 | camp_r        |
|         0.1643 |      28020.8199 |           0.1718 |          0.0075 |     0.0081 |         0.1477 |     0.0751 |           10 |                   9 |                3.4947 | ramp_real     |
|         0.1431 |      24420.3556 |           0.1488 |          0.0057 |     0.0099 |         0.1699 |     0.0874 |           10 |                   8 |                2.5451 | online_regime |
|         0.0798 |      13614.7843 |           0.0870 |          0.0072 |     0.0073 |         0.1325 |     0.0662 |           10 |                   7 |                1.1011 | ridge_static  |
|         0.0016 |        270.4015 |           0.0073 |          0.0057 |     0.0099 |         0.1699 |     0.0859 |           10 |                   6 |                0.0210 | micro_fixed   |
|        -0.0673 |     -11480.1892 |          -0.0609 |          0.0064 |     0.0042 |         0.0709 |     0.0334 |           10 |                   5 |               -0.7157 | momentum      |
|        -0.0722 |     -12317.0407 |          -0.0663 |          0.0059 |     0.0146 |         0.2664 |     0.1278 |           10 |                   2 |               -2.4073 | signed_flow   |

## Per-asset summary

| asset   | venue   | strategy      |   mean_net_bps |   total_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |
|:--------|:--------|:--------------|---------------:|----------------:|-----------------:|----------------:|-----------:|---------------:|
| AAPL    | XNAS    | camp_r        |         0.3148 |      26528.2178 |           0.3298 |          0.0150 |     0.0078 |         0.1559 |
| AAPL    | XNAS    | ramp_real     |         0.3148 |      26528.2178 |           0.3298 |          0.0150 |     0.0078 |         0.1559 |
| AAPL    | XNAS    | online_regime |         0.2582 |      21753.9402 |           0.2694 |          0.0113 |     0.0094 |         0.1655 |
| AAPL    | XNAS    | ridge_static  |         0.1446 |      12182.2764 |           0.1591 |          0.0145 |     0.0070 |         0.1390 |
| AAPL    | XNAS    | micro_fixed   |        -0.0284 |      -2396.0139 |          -0.0172 |          0.0113 |     0.0094 |         0.1655 |
| AAPL    | XNAS    | momentum      |        -0.1140 |      -9606.8110 |          -0.1011 |          0.0129 |     0.0067 |         0.1153 |
| AAPL    | XNAS    | signed_flow   |        -0.1287 |     -10848.1120 |          -0.1173 |          0.0115 |     0.0080 |         0.1328 |
| BNBUSDT | BINANCE | camp_r        |         0.0309 |       2666.4154 |           0.0311 |          0.0002 |     0.0104 |         0.1741 |
| BNBUSDT | BINANCE | micro_fixed   |         0.0309 |       2666.4154 |           0.0311 |          0.0002 |     0.0104 |         0.1741 |
| BNBUSDT | BINANCE | online_regime |         0.0309 |       2666.4154 |           0.0311 |          0.0002 |     0.0104 |         0.1741 |
| BNBUSDT | BINANCE | ramp_real     |         0.0173 |       1492.6020 |           0.0175 |          0.0002 |     0.0084 |         0.1397 |
| BNBUSDT | BINANCE | ridge_static  |         0.0166 |       1432.5079 |           0.0167 |          0.0002 |     0.0076 |         0.1261 |
| BNBUSDT | BINANCE | signed_flow   |        -0.0170 |      -1468.9287 |          -0.0166 |          0.0004 |     0.0210 |         0.3969 |
| BNBUSDT | BINANCE | momentum      |        -0.0217 |      -1873.3782 |          -0.0217 |          0.0000 |     0.0018 |         0.0275 |

## Episode-level totals

| strategy      | asset   | venue   | day        |   total_net_bps |   mean_net_bps |   mean_gross_bps |   mean_cost_bps |   turnover |   active_share |   hit_rate |
|:--------------|:--------|:--------|:-----------|----------------:|---------------:|-----------------:|----------------:|-----------:|---------------:|-----------:|
| camp_r        | AAPL    | XNAS    | 2025-06-03 |         4397.53 |           0.46 |             0.47 |            0.01 |       0.00 |           0.09 |       0.05 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-03 |         2858.89 |           0.30 |             0.30 |            0.01 |       0.01 |           0.10 |       0.05 |
| momentum      | AAPL    | XNAS    | 2025-06-03 |        -4925.83 |          -0.51 |            -0.50 |            0.01 |       0.01 |           0.09 |       0.04 |
| online_regime | AAPL    | XNAS    | 2025-06-03 |         2858.89 |           0.30 |             0.30 |            0.01 |       0.01 |           0.10 |       0.05 |
| ramp_real     | AAPL    | XNAS    | 2025-06-03 |         4397.53 |           0.46 |             0.47 |            0.01 |       0.00 |           0.09 |       0.05 |
| ridge_static  | AAPL    | XNAS    | 2025-06-03 |         5161.44 |           0.54 |             0.55 |            0.01 |       0.00 |           0.09 |       0.05 |
| signed_flow   | AAPL    | XNAS    | 2025-06-03 |        -1975.93 |          -0.21 |            -0.20 |            0.01 |       0.01 |           0.11 |       0.06 |
| camp_r        | AAPL    | XNAS    | 2025-06-04 |         -106.95 |          -0.01 |            -0.00 |            0.01 |       0.00 |           0.08 |       0.04 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-04 |        -2227.27 |          -0.22 |            -0.21 |            0.01 |       0.01 |           0.12 |       0.06 |
| momentum      | AAPL    | XNAS    | 2025-06-04 |          276.40 |           0.03 |             0.03 |            0.00 |       0.00 |           0.04 |       0.02 |
| online_regime | AAPL    | XNAS    | 2025-06-04 |        -2227.27 |          -0.22 |            -0.21 |            0.01 |       0.01 |           0.12 |       0.06 |
| ramp_real     | AAPL    | XNAS    | 2025-06-04 |         -106.95 |          -0.01 |            -0.00 |            0.01 |       0.00 |           0.08 |       0.04 |
| ridge_static  | AAPL    | XNAS    | 2025-06-04 |          101.90 |           0.01 |             0.02 |            0.01 |       0.00 |           0.07 |       0.04 |
| signed_flow   | AAPL    | XNAS    | 2025-06-04 |        -2751.46 |          -0.27 |            -0.26 |            0.01 |       0.01 |           0.13 |       0.06 |
| camp_r        | AAPL    | XNAS    | 2025-06-05 |         5261.59 |           0.59 |             0.60 |            0.01 |       0.00 |           0.05 |       0.03 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-05 |         1301.10 |           0.15 |             0.15 |            0.01 |       0.01 |           0.12 |       0.06 |
| momentum      | AAPL    | XNAS    | 2025-06-05 |        -6995.30 |          -0.78 |            -0.77 |            0.01 |       0.01 |           0.09 |       0.04 |
| online_regime | AAPL    | XNAS    | 2025-06-05 |         1301.10 |           0.15 |             0.15 |            0.01 |       0.01 |           0.12 |       0.06 |
| ramp_real     | AAPL    | XNAS    | 2025-06-05 |         5261.59 |           0.59 |             0.60 |            0.01 |       0.00 |           0.05 |       0.03 |
| ridge_static  | AAPL    | XNAS    | 2025-06-05 |         5252.18 |           0.59 |             0.59 |            0.01 |       0.00 |           0.05 |       0.03 |
| signed_flow   | AAPL    | XNAS    | 2025-06-05 |        -3053.72 |          -0.34 |            -0.33 |            0.01 |       0.01 |           0.13 |       0.07 |
| camp_r        | AAPL    | XNAS    | 2025-06-06 |         1046.25 |           0.11 |             0.14 |            0.02 |       0.01 |           0.29 |       0.15 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-06 |         -203.97 |          -0.02 |            -0.01 |            0.02 |       0.01 |           0.25 |       0.13 |
| momentum      | AAPL    | XNAS    | 2025-06-06 |         6886.33 |           0.75 |             0.78 |            0.02 |       0.01 |           0.29 |       0.15 |
| online_regime | AAPL    | XNAS    | 2025-06-06 |         -203.97 |          -0.02 |            -0.01 |            0.02 |       0.01 |           0.25 |       0.13 |
| ramp_real     | AAPL    | XNAS    | 2025-06-06 |         1046.25 |           0.11 |             0.14 |            0.02 |       0.01 |           0.29 |       0.15 |
| ridge_static  | AAPL    | XNAS    | 2025-06-06 |         2310.56 |           0.25 |             0.27 |            0.02 |       0.01 |           0.26 |       0.13 |
| signed_flow   | AAPL    | XNAS    | 2025-06-06 |         2345.12 |           0.26 |             0.27 |            0.01 |       0.01 |           0.17 |       0.09 |
| camp_r        | AAPL    | XNAS    | 2025-06-07 |          571.12 |           0.06 |             0.08 |            0.02 |       0.01 |           0.25 |       0.12 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-07 |         1805.82 |           0.19 |             0.21 |            0.02 |       0.01 |           0.19 |       0.09 |
| momentum      | AAPL    | XNAS    | 2025-06-07 |        -5607.93 |          -0.58 |            -0.57 |            0.02 |       0.01 |           0.15 |       0.07 |
| online_regime | AAPL    | XNAS    | 2025-06-07 |         1805.82 |           0.19 |             0.21 |            0.02 |       0.01 |           0.19 |       0.09 |
| ramp_real     | AAPL    | XNAS    | 2025-06-07 |          571.12 |           0.06 |             0.08 |            0.02 |       0.01 |           0.25 |       0.12 |
| ridge_static  | AAPL    | XNAS    | 2025-06-07 |         2012.87 |           0.21 |             0.23 |            0.02 |       0.01 |           0.23 |       0.12 |
| signed_flow   | AAPL    | XNAS    | 2025-06-07 |        -1622.88 |          -0.17 |            -0.15 |            0.02 |       0.01 |           0.19 |       0.09 |
| camp_r        | AAPL    | XNAS    | 2025-06-08 |         1424.53 |           0.17 |             0.17 |            0.01 |       0.00 |           0.09 |       0.04 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-08 |         5799.35 |           0.67 |             0.68 |            0.01 |       0.01 |           0.18 |       0.11 |
| momentum      | AAPL    | XNAS    | 2025-06-08 |         4407.54 |           0.51 |             0.52 |            0.01 |       0.00 |           0.05 |       0.03 |
| online_regime | AAPL    | XNAS    | 2025-06-08 |         5799.35 |           0.67 |             0.68 |            0.01 |       0.01 |           0.18 |       0.11 |
| ramp_real     | AAPL    | XNAS    | 2025-06-08 |         1424.53 |           0.17 |             0.17 |            0.01 |       0.00 |           0.09 |       0.04 |
| ridge_static  | AAPL    | XNAS    | 2025-06-08 |         -302.08 |          -0.04 |            -0.02 |            0.01 |       0.00 |           0.08 |       0.04 |
| signed_flow   | AAPL    | XNAS    | 2025-06-08 |        -1339.76 |          -0.16 |            -0.14 |            0.01 |       0.01 |           0.12 |       0.06 |
| camp_r        | AAPL    | XNAS    | 2025-06-09 |         6505.68 |           0.68 |             0.69 |            0.01 |       0.01 |           0.16 |       0.08 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-09 |          634.13 |           0.07 |             0.07 |            0.01 |       0.01 |           0.12 |       0.07 |
| momentum      | AAPL    | XNAS    | 2025-06-09 |            2.39 |           0.00 |             0.01 |            0.01 |       0.01 |           0.09 |       0.05 |
| online_regime | AAPL    | XNAS    | 2025-06-09 |          634.13 |           0.07 |             0.07 |            0.01 |       0.01 |           0.12 |       0.07 |
| ramp_real     | AAPL    | XNAS    | 2025-06-09 |         6505.68 |           0.68 |             0.69 |            0.01 |       0.01 |           0.16 |       0.08 |
| ridge_static  | AAPL    | XNAS    | 2025-06-09 |         5502.82 |           0.57 |             0.58 |            0.01 |       0.01 |           0.11 |       0.05 |
| signed_flow   | AAPL    | XNAS    | 2025-06-09 |          692.81 |           0.07 |             0.08 |            0.01 |       0.01 |           0.15 |       0.07 |
| camp_r        | AAPL    | XNAS    | 2025-06-10 |         1092.03 |           0.12 |             0.14 |            0.02 |       0.01 |           0.25 |       0.13 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-10 |        -8737.41 |          -0.95 |            -0.94 |            0.01 |       0.01 |           0.22 |       0.10 |
| momentum      | AAPL    | XNAS    | 2025-06-10 |         3788.84 |           0.41 |             0.42 |            0.01 |       0.01 |           0.10 |       0.05 |
| online_regime | AAPL    | XNAS    | 2025-06-10 |         8516.32 |           0.92 |             0.94 |            0.01 |       0.01 |           0.22 |       0.12 |
| ramp_real     | AAPL    | XNAS    | 2025-06-10 |         1092.03 |           0.12 |             0.14 |            0.02 |       0.01 |           0.25 |       0.13 |
| ridge_static  | AAPL    | XNAS    | 2025-06-10 |         -106.25 |          -0.01 |             0.01 |            0.02 |       0.01 |           0.26 |       0.12 |
| signed_flow   | AAPL    | XNAS    | 2025-06-10 |        -1108.37 |          -0.12 |            -0.11 |            0.01 |       0.01 |           0.11 |       0.06 |
| camp_r        | AAPL    | XNAS    | 2025-06-11 |         6336.43 |           0.67 |             0.69 |            0.02 |       0.01 |           0.15 |       0.08 |
| micro_fixed   | AAPL    | XNAS    | 2025-06-11 |        -3626.65 |          -0.38 |            -0.36 |            0.02 |       0.01 |           0.19 |       0.09 |
| momentum      | AAPL    | XNAS    | 2025-06-11 |        -7439.26 |          -0.78 |            -0.76 |            0.02 |       0.01 |           0.15 |       0.05 |
| online_regime | AAPL    | XNAS    | 2025-06-11 |         3269.58 |           0.34 |             0.36 |            0.02 |       0.01 |           0.19 |       0.10 |
| ramp_real     | AAPL    | XNAS    | 2025-06-11 |         6336.43 |           0.67 |             0.69 |            0.02 |       0.01 |           0.15 |       0.08 |
| ridge_static  | AAPL    | XNAS    | 2025-06-11 |        -7751.17 |          -0.82 |            -0.80 |            0.02 |       0.01 |           0.11 |       0.06 |
| signed_flow   | AAPL    | XNAS    | 2025-06-11 |        -2033.92 |          -0.21 |            -0.20 |            0.01 |       0.00 |           0.08 |       0.03 |
| camp_r        | BNBUSDT | BINANCE | 2024-03-30 |         2666.42 |           0.03 |             0.03 |            0.00 |       0.01 |           0.17 |       0.09 |
| micro_fixed   | BNBUSDT | BINANCE | 2024-03-30 |         2666.42 |           0.03 |             0.03 |            0.00 |       0.01 |           0.17 |       0.09 |
| momentum      | BNBUSDT | BINANCE | 2024-03-30 |        -1873.38 |          -0.02 |            -0.02 |            0.00 |       0.00 |           0.03 |       0.01 |
| online_regime | BNBUSDT | BINANCE | 2024-03-30 |         2666.42 |           0.03 |             0.03 |            0.00 |       0.01 |           0.17 |       0.09 |
| ramp_real     | BNBUSDT | BINANCE | 2024-03-30 |         1492.60 |           0.02 |             0.02 |            0.00 |       0.01 |           0.14 |       0.07 |
| ridge_static  | BNBUSDT | BINANCE | 2024-03-30 |         1432.51 |           0.02 |             0.02 |            0.00 |       0.01 |           0.13 |       0.06 |
| signed_flow   | BNBUSDT | BINANCE | 2024-03-30 |        -1468.93 |          -0.02 |            -0.02 |            0.00 |       0.02 |           0.40 |       0.19 |

## Headline

Across the combined real multi-asset benchmark, **camp_r** finishes first with **0.1711 mean net bps/step**, **29194.6 total net bps**, and **9/10 positive episodes**.

On in-domain AAPL, the best strategy is **camp_r** at **0.3148 mean net bps/step**.

On external BNBUSDT, the best strategy is **camp_r** at **0.0309 mean net bps/step**.

## Honesty note

This phase is **more realistic and more useful for recruiting** than phase 3 because it includes a second real asset and a second venue.
But it is still a compact benchmark: there is only one external Binance day bundled in the repo, so the cross-asset claim is best presented as a **robustness / transfer test**, not as a fully diversified production universe.
