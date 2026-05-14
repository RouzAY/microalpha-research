# Benchmark summary

## Experimental setup

- Symbol: AAPL
- Train days: 2025-06-02, 2025-06-03, 2025-06-04, 2025-06-05, 2025-06-06, 2025-06-07
- Validation days: 2025-06-08, 2025-06-09
- Test days: 2025-06-10, 2025-06-11
- Parent order sizes: 1000, 3000, 5000
- Window length: 900 book updates
- MALP weights selected on validation set: liquidity=0.800, alpha=0.800

## Held-out real-data results

|   qty | policy   |   mean_is_bps |
|------:|:---------|--------------:|
|  1000 | alpha    |         2.733 |
|  1000 | liq      |         3.043 |
|  1000 | malp     |         2.158 |
|  1000 | twap     |         4.917 |
|  1000 | vwap     |         4.748 |
|  3000 | alpha    |         2.875 |
|  3000 | liq      |         3.113 |
|  3000 | malp     |         2.215 |
|  3000 | twap     |         5.089 |
|  3000 | vwap     |         5.007 |
|  5000 | alpha    |         2.982 |
|  5000 | liq      |         3.149 |
|  5000 | malp     |         2.251 |
|  5000 | twap     |         5.202 |
|  5000 | vwap     |         5.188 |

## Synthetic stress-test results

|   qty | policy   |   mean_is_bps |
|------:|:---------|--------------:|
|  1000 | alpha    |         3.192 |
|  1000 | liq      |         3.024 |
|  1000 | malp     |         2.137 |
|  1000 | twap     |         5.617 |
|  1000 | vwap     |         5.617 |
|  3000 | alpha    |         3.446 |
|  3000 | liq      |         3.105 |
|  3000 | malp     |         2.223 |
|  3000 | twap     |         5.857 |
|  3000 | vwap     |         5.857 |
|  5000 | alpha    |         3.629 |
|  5000 | liq      |         3.147 |
|  5000 | malp     |         2.268 |
|  5000 | twap     |         6.019 |
|  5000 | vwap     |         6.019 |
