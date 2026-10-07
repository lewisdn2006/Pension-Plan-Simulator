# Monte Carlo Engine

One geometric Brownian motion (GBM) path simulator used for two jobs: pricing European, Asian and
barrier options, and forecasting the outcome of a pension paid into a share index. The two jobs
used to be separate scripts, each with its own copy of the same simulation. They now share one
function, `mcengine.gbm.simulate_gbm_paths`, and the only difference between them is the drift
that the caller passes in.

| Job | Drift passed to the simulator | What the paths are used for |
|---|---|---|
| Option pricing | risk-neutral, r - q | the discounted average payoff is the option price |
| Pension forecast | real-world growth rate, mu | the spread of final funds is the spread of possible outcomes |

## What it does

- **Options** (`mcengine/options.py`): European calls and puts with control variate, antithetic
  sampling and Greeks; arithmetic Asian options with a geometric-average control variate; barrier
  options (up or down, in or out).
- **Black-Scholes** (`mcengine/black_scholes.py`): closed-form prices and Greeks, used as the
  check on the European results, and the closed-form geometric Asian price.
- **Implied volatility** (`mcengine/implied_vol.py`): Brent's method, with an error instead of a
  made-up answer when no volatility fits the price.
- **Pension model** (`mcengine/pension.py`): a fixed amount paid in every month, invested in a share
  index. It gives the chance of finishing below what was paid in, above double it, or above
  2 million, and the monthly payment needed for a 95% chance of reaching a target fund.

## Results

Every number below is produced by `scripts/measure_results.py` and is in
[results/measured_results_full.md](results/measured_results_full.md) (and
[results/measured_results.md](results/measured_results.md), which uses 10,000 pension paths
instead of 100,000 so that it runs in under 15 seconds). All random numbers are seeded.

**European call** (S = 100, K = 105, one year, r = 5%, q = 2%, volatility 25%; Black-Scholes price
8.9412). 400 independent runs of 20,000 paths each:

| mode | mean price | SD across runs | SD smaller by | 95% interval covers exact |
|---|---|---|---|---|
| plain | 8.9448 | 0.1131 | 1.00x | 95.8% |
| antithetic only | 8.9417 | 0.0960 | 1.18x | 94.2% |
| control variate only | 8.9405 | 0.0524 | 2.16x | 95.8% |
| antithetic + control variate (default) | 8.9432 | 0.0232 | 4.87x | 92.0% |

The error shrinks with the square root of the number of paths: the fitted slope of log error
against log paths is -0.53, against a theoretical -0.50. The finite-difference Greeks agree with
Black-Scholes: delta 0.5095 against 0.5096, gamma 0.01561 against 0.01562, vega 39.16 against
39.06 (400,000 paths).

**Asian call** (average of 52 weekly prices, strike 100, 200,000 paths): the geometric control
variate cuts the standard error from 0.0216 to 0.00074, a factor of 29. The simulated geometric
Asian price (6.0962, standard error 0.0208) matches its closed form (6.0747).

**Barrier** (up-and-out call, barrier 120, strike 100): 0.9212 with 52 monitoring dates and
0.7345 with 1,000. The price falls as the barrier is checked more often because fewer crossings
are missed. A knock-in plus the matching knock-out always equals the vanilla option on the same
paths, and the tests check that exactly.

**Pension** (40 years, 5% growth, 1,000 a month, 100,000 paths): a 6.4% chance of finishing with
less than the 480,000 paid in, 63.8% of finishing with more than double it, and 22.4% of finishing
above 2 million. For a 95% chance of at least 1 million, the monthly payment needed is:

| growth rate | 40 years | 20 years | extra for starting 20 years later |
|---|---|---|---|
| 3% | 3,320 | 6,280 | 2,960 |
| 5% | 2,260 | 5,200 | 2,940 |
| 7% | 1,460 | 4,260 | 2,800 |

The whole pension analysis, including the six probability curves, takes 3.9 seconds on a laptop
for 100,000 paths. The pension code gives the same paths as the original script for the same seed
(largest difference 0) and the same final funds to within 6e-09
([results/comparison_with_original.txt](results/comparison_with_original.txt)).

## How it works

**The simulator.** Each step uses the exact solution of the GBM equation,
`S(t+dt) = S(t) * exp((drift - sigma^2/2) dt + sigma sqrt(dt) Z)`, not the Euler step. The exact step
has no discretisation error, so the final price has the same distribution for any number of
steps, and the price cannot go negative. That is why a European option is priced with one step.

**Standard errors.** With antithetic sampling the two paths of a pair are correlated, so they are
averaged first and the standard error is computed from the independent pair averages. The
control variates (the final price for a European option, the geometric average for an Asian one)
have exactly known means. The weight on the control is the regression slope
`cov(payoff, control) / var(control)`, estimated from the same sample.

**Greeks.** Delta, gamma and vega are finite differences of the Monte Carlo price. Every bumped
price uses the same random numbers as the base price. With fresh random numbers per bump the noise
in each price is larger than the change being measured, and the result is meaningless.

**Pension shortcut.** Each month M buys M/S shares, so the final fund is exactly M times the
fund for a payment of 1. One simulation therefore gives the fund for every payment amount, and
the chance of beating a target for each amount is a lookup in a sorted array.

## Running it

```
pip install -r requirements.txt
python examples/price_options.py            # options, plots in examples/output/
python examples/pension_outcomes.py         # pension, 10,000 paths; add --paths 100000 for the full size
python scripts/measure_results.py           # rewrites results/measured_results.md (--full: 100,000 pension paths)
```

```python
from mcengine.options import MarketData, MonteCarloEngine, OptionContract

market = MarketData(spot_price=100, risk_free_rate=0.05, dividend_yield=0.02, volatility=0.25)
engine = MonteCarloEngine(seed=42)
result = engine.price_european(market, OptionContract("call", strike=105, expiry=1.0))
print(result.price, result.std_error, result.delta)
```

## Tests

```
python -m pytest
```

95 tests pass and 4 are skipped (the skips are implied-volatility cases where the price barely
depends on volatility, so the volatility cannot be recovered). They check, among other things:
the simulator's mean, variance and independence from the number of steps; Black-Scholes against
textbook values, put-call parity and finite differences of its own price; every pricing mode
against Black-Scholes within four standard errors; that the reported 95% interval covers the true
price about 95% of the time; knock-in plus knock-out equals vanilla; and the pension fund against a
payment-by-payment calculation.

## Limitations

- Volatility, interest rate and dividend yield are constants. There is no volatility smile, so
  implied volatilities from real quotes will differ by strike.
- Barriers are checked at the simulated times only. Crossings between two times are missed, which
  makes knock-outs worth slightly more and knock-ins slightly less than continuous monitoring.
- American options are not implemented.
- The 95% interval from the combined antithetic and control-variate estimator covered the true
  price 92.0% of the time in 400 runs, and its reported standard error was 4% below the spread
  of the prices. It is slightly too narrow.
- The pension model has a constant growth rate and volatility (15%), a fixed monthly payment and
  no inflation, fees or tax. It pays at months 0 to 12T inclusive, 12T + 1 payments, as the original
  did. The growth rate is an assumption, and the answer depends on it strongly (see the table).
- Probabilities from n paths have a sampling error of about `sqrt(p(1-p)/n)`: 0.0015 at p = 0.5
  with 100,000 paths. The payment needed is read off a grid in steps of 20.

## Related projects

- [self-hedging-gold-vault](https://github.com/lewisdn2006/self-hedging-gold-vault): a gold holding
  that hedges itself through a prediction market with simulated traders.
- [trading-backtester](https://github.com/lewisdn2006/trading-backtester): one cost-aware
  backtester for any trading signal, with a replay window.

[CHANGES.md](CHANGES.md) lists what changed from the two original scripts.
