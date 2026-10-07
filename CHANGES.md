# What changed from the two original scripts

This project combines `option_pricer.py` (an option pricer) and `pension_simulator.py` (a Monte
Carlo pension model). Both simulated geometric Brownian motion with their own code. The faults
below were measured on the original scripts by `scripts/compare_with_original.py`; the output is
saved in [results/comparison_with_original.txt](results/comparison_with_original.txt).

## Shared

- One GBM simulator (`mcengine/gbm.py`) replaces the two copies. For the same seed it gives
  exactly the same pension paths as the original (largest difference 0).
- The code is split into modules, with 95 tests, and plots are saved to files instead of opening
  windows.

## Option pricer: faults fixed

| Fault in the original | Measured effect | Now |
|---|---|---|
| Knock-in barrier options were priced as knock-outs. | Up-and-in and up-and-out both priced at 0.8043 (barrier 120, same seed). | Up-and-out 0.7945, up-and-in 10.2666. Their sum equals the vanilla option, and a test checks it. |
| Standard error treated the two paths of an antithetic pair as independent. | Reported standard error was 0.72 of the true spread, and the 95% interval covered the true price 83.0% of the time (200 runs). | Standard error computed from pair averages: ratio 0.96 to 1.00, coverage 92.0% to 95.8% (400 runs). |
| Each Greek used fresh random numbers for every bump. | At 100,000 paths, gamma was between -0.75 and +0.61 across five seeds (exact 0.0156), and delta between 0.35 and 0.50 (exact 0.5096). | Common random numbers: gamma 0.0155 to 0.0158, delta 0.5073 to 0.5101. |
| Monte Carlo vega was per 1.00 of volatility; Black-Scholes vega was per 0.01. | The two printed values differed by a factor of 100. | Both per 1.00. |
| Implied volatility fell back to a clipped Newton search when no volatility fitted. | A price of 1.00 for a call worth at least 54.88 returned a volatility of 0.0010. | Raises an error stating the range of prices that can be matched. |
| The Asian average included the starting price. | With 252 steps it averaged 253 prices, the first of which is not a fixing date. | Averages the fixing dates only. |
| European options were simulated with 252 steps. | Slow, and no more accurate. | One step. The exact simulation gives the same distribution. |

Also removed: global `np.random.seed`, which changed the random state for the whole program; a
blocking `plt.show()` inside the pricing function; a blanket `warnings.filterwarnings('ignore')`;
CSV loaders that nothing called; an unused `scipy.optimize.minimize` import; a computed but unused
control price; and the `style` field, which duplicated the barrier fields. The docstring said
American options were supported. They were not, and this is now stated.

Added: a geometric-average control variate for Asian options (standard error 29 times smaller),
a closed-form geometric Asian price to check it, and input validation.

## Pension simulator: what changed

- The probability curve recomputed the fund on every path for every payment amount. The fund is
  proportional to the payment, so it is now computed once. On 2,000 paths and 200 payments this
  was 176 times faster in the same run (0.06 s against 0.0004 s), with identical probabilities.
  The full analysis at 100,000 paths now takes 3.9 seconds.
- `observed_probability` took a growth-rate argument it never used. Removed.
- The model, the 15% volatility and the 12T + 1 payment convention are unchanged.
