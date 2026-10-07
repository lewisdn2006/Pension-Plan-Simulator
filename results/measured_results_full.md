# Measured results

Produced by `scripts/measure_results.py --full` with Python 3.9.6, numpy 2.0.2. All random numbers are seeded.

## 1. European call: accuracy, variance reduction and the reported error

European call, S = 100, K = 105, T = 1 year, r = 5%, q = 2%, volatility 25%. Black-Scholes price 8.9412. Each row is 400 independent runs of 20,000 paths (seeds 0 to 399). The simulation is exact, so one time step per path.

| mode | mean price | mean - exact | SD across runs | SD smaller by | mean reported SE | reported SE / SD | 95% CI covers exact |
|---|---|---|---|---|---|---|---|
| plain | 8.9448 | +0.0036 | 0.1131 | 1.00x | 0.1132 | 1.00 | 95.8% |
| antithetic only | 8.9417 | +0.0005 | 0.0960 | 1.18x | 0.0939 | 0.98 | 94.2% |
| control variate only | 8.9405 | -0.0007 | 0.0524 | 2.16x | 0.0526 | 1.00 | 95.8% |
| antithetic + control variate | 8.9432 | +0.0020 | 0.0232 | 4.87x | 0.0223 | 0.96 | 92.0% |

## 2. Convergence

Root-mean-square error against Black-Scholes over 50 runs at each size (control variate on).

| paths | RMS error |
|---|---|
| 1,000 | 0.09815 |
| 10,000 | 0.03160 |
| 100,000 | 0.00937 |
| 1,000,000 | 0.00259 |

Fitted slope of log(error) against log(paths): -0.53 (theory: -0.50).

## 3. Greeks (finite differences with common random numbers)

400,000 paths, seed 8.

| Greek | Monte Carlo | Black-Scholes |
|---|---|---|
| delta | 0.5095 | 0.5096 |
| gamma | 0.01561 | 0.01562 |
| vega (per 1.00 of volatility) | 39.16 | 39.06 |

## 4. Asian call (average of 52 weekly prices)

200,000 paths.

| quantity | price | standard error |
|---|---|---|
| geometric Asian, closed form | 6.0747 | - |
| geometric Asian, simulated | 6.0962 | 0.0208 |
| arithmetic Asian, plain | 6.3760 | 0.0216 |
| arithmetic Asian, geometric control variate | 6.3690 | 0.00074 |

The control variate makes the standard error 29 times smaller.

## 5. Barrier options (up barrier 120, call strike 100)

200,000 paths. Vanilla call (Black-Scholes): 11.1238.

| barrier checks | up-and-out | up-and-in | sum |
|---|---|---|---|
| 12 | 1.2218 | 9.8829 | 11.1047 |
| 52 | 0.9212 | 10.1560 | 11.0772 |
| 252 | 0.7968 | 10.2969 | 11.0937 |
| 1000 | 0.7345 | 10.4053 | 11.1397 |

The sum is the vanilla price on the same paths. The knock-out falls and the knock-in rises as the barrier is checked more often, because fewer crossings are missed.

## 6. Implied volatility round trip

472 options (calls and puts, strikes 70 to 140, volatilities 5% to 150%): largest error in the recovered volatility 3.7e-12.

## 7. Pension model

100,000 simulated index paths per case.

**40 years, 5% growth, 1,000 a month (480,000 paid in)**

| outcome | probability |
|---|---|
| finish with less than was paid in | 0.0643 |
| finish with more than double what was paid in | 0.6378 |
| finish with more than 2 million | 0.2244 |

**Monthly payment (in steps of 20) for a 95% chance of at least 1 million**

| growth rate | 40 years | 20 years | extra for starting 20 years later |
|---|---|---|---|
| 3% | 3,320 | 6,280 | 2,960 |
| 5% | 2,260 | 5,200 | 2,940 |
| 7% | 1,460 | 4,260 | 2,800 |

All of the above took 3.9 seconds.

**Checking the shortcut and timing it.** The original code recomputed the fund on every path for every payment amount. This package computes the fund for a payment of 1 once and scales it. Both give the same probabilities (compared below on 2,000 paths and 200 payments).

Largest difference between the two probability curves: 0.0e+00. Payment-by-payment: 0.049 s. Scaled: 0.0003 s (164 times faster).

Total run time: 16 seconds.
