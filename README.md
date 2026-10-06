# Pension Plan Simulator

A Monte Carlo simulation of a pension that is paid into every month and invested in a share index. It simulates 100,000 possible futures for the stock market and reports the whole range of outcomes, instead of one forecast based on an average return.

## The model

The share index follows geometric Brownian motion:

```
dS = mu * S dt + sigma * S dW
```

- `mu` is the average annual growth rate (3%, 5% and 7% are tested).
- `sigma` is the annual volatility, set to 15%.
- Each path is simulated in monthly steps using the exact solution `S(n+1) = S(n) * exp((mu - sigma²/2) dt + sigma dW)`, which keeps the price positive on every path. The simpler step `S + mu S dt + sigma S dW` can go negative.

Each month the contribution buys shares at that month's price. The pension's final value is the number of shares held multiplied by the final price. It does not depend on the starting share price, because that cancels between the number of shares bought and the final price.

## What it answers

For someone paying in £1,000 a month for 40 years with 5% average growth, one run of 100,000 paths gave:

| Outcome | Probability |
|---|---|
| Finishing with less than was paid in | 6.7% |
| At least doubling the money paid in | 63.5% |
| Finishing with more than £2 million | 22.4% |

Monthly contribution needed for a 95% chance of a £1 million fund (tested in steps of £20):

| Average growth | Starting 40 years before retirement | Starting 20 years before |
|---|---|---|
| 3% | £3,320 | £6,260 |
| 5% | £2,240 | £5,180 |
| 7% | £1,440 | £4,240 |

Starting 20 years later costs roughly £2,800 to £2,900 more every month for the same level of confidence. Results change slightly from run to run because the paths are random.

It also plots example paths with the mean and the 5th and 95th percentiles, a histogram of final pension values, and the probability of reaching the target against the monthly contribution.

## Running it

```bash
pip install numpy matplotlib
python pension_simulator.py
```

A full run takes several minutes and over 1 GB of memory, most of it spent testing 200 to 500 contribution levels against 100,000 paths. Close each plot window to move on to the next step.

The functions can also be used on their own:

```python
from pension_simulator import simulate_gbm, pension_value

paths = simulate_gbm(mu=0.05, P=10000, T=40, N=480, seed=1)   # 10,000 paths, monthly steps
values = pension_value(paths, T=40, M=500)                     # £500 a month for 40 years
```

## Assumptions and limits

- Growth rate and volatility are constant, and log returns are normally distributed. Real markets have fatter tails and changing volatility.
- No fees, taxes, inflation or salary growth. All amounts are in today's money only if `mu` is read as a real return.
- Everything is invested in one index, with no move into safer assets close to retirement.

## Built with

Python, NumPy, matplotlib.
