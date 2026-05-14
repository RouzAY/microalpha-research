# Phase 2 alpha researcher benchmark

## Setup

- Real calibration source: AAPL LOB snapshots from the bundled sample data
- Synthetic train/val/test episodes: 30 / 10 / 12
- Synthetic symbols per episode: 10
- Steps per episode: 900
- Portfolio is dollar-neutral with explicit turnover costs
- Our method: RAMP = Regime-Adaptive Multi-horizon Portfolio

## Strategy table

| strategy   |   mean_gross_bps |   mean_net_bps |   total_net_bps |   hit_rate |   mean_turnover |   sharpe_like |
|:-----------|-----------------:|---------------:|----------------:|-----------:|----------------:|--------------:|
| ramp       |           0.6282 |         0.6027 |       6509.3406 |     0.9532 |          0.1808 |        1.1037 |
| uniform    |           0.5584 |         0.4746 |       5126.1823 |     0.9188 |          0.5920 |        1.0810 |
| ridge5     |           0.5190 |         0.4181 |       4515.7721 |     0.8447 |          0.6809 |        0.7410 |
| imbalance  |           0.3730 |         0.2884 |       3114.7314 |     0.8421 |          0.6559 |        0.8872 |
| momentum   |           0.2283 |         0.1954 |       2110.8067 |     0.6956 |          0.2235 |        0.3655 |

## Headline

RAMP finishes first with **0.6027 mean net bps/step** and **6509.3 total net bps** on the held-out synthetic test universe.
