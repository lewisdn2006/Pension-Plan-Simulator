"""A pension paid into every month and invested in a share index that follows GBM.

Each month a fixed amount buys shares at that month's price. The final fund is the number of
shares held times the final price. Simulating many index paths gives the whole distribution
of outcomes, which answers questions such as the chance of finishing with less than was paid in
and the monthly amount needed for a 95% chance of reaching a target.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .gbm import simulate_gbm_paths

SIGMA = 0.15
S0 = 1.0
MONTHS_PER_YEAR = 12


def simulate_index_paths(
    mu: float, n_paths: int, years: int, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Monthly index paths, shape (n_paths, 12 * years + 1), with expected growth rate mu."""
    return simulate_gbm_paths(S0, mu, SIGMA, years, MONTHS_PER_YEAR * years, n_paths, rng=rng)


def unit_fund_values(paths: np.ndarray, years: int) -> np.ndarray:
    """Final fund on every path for a payment of 1 a month, paid at months 0, 1, ..., 12 * years.

    The fund is exactly proportional to the monthly payment M, because M buys M / S shares
    each month. The fund for any other payment is M times this array, so one simulation can
    be reused for every payment amount. It does not depend on the starting price either:
    every price is S0 times an exponential, so S0 cancels between shares bought and the
    final price.
    """
    last = MONTHS_PER_YEAR * years
    if last >= paths.shape[1]:
        raise ValueError("The paths are shorter than the number of years asked for.")
    shares_per_unit = np.sum(1.0 / paths[:, : last + 1], axis=1)
    return shares_per_unit * paths[:, last]


def fund_values(paths: np.ndarray, years: int, monthly: float) -> np.ndarray:
    """Final fund on every path for a payment of `monthly` a month."""
    if monthly < 0:
        raise ValueError("Monthly payment must not be negative.")
    return monthly * unit_fund_values(paths, years)


def outcome_probabilities(
    values: np.ndarray, years: int, monthly: float, high_target: float = 2_000_000
) -> Tuple[float, float, float]:
    """Chance of finishing with less than was paid in, with more than double, and above high_target.

    "Paid in" is 12 * years * monthly, as in the original model.
    """
    values = np.asarray(values)
    paid_in = MONTHS_PER_YEAR * years * monthly
    return (
        float(np.mean(values < paid_in)),
        float(np.mean(values > 2 * paid_in)),
        float(np.mean(values > high_target)),
    )


def probability_curve(unit_values: np.ndarray, payments: Sequence[float], target: float) -> np.ndarray:
    """Chance that the fund finishes above `target`, for each monthly payment in `payments`.

    The fund is payment * unit_values, so it exceeds the target exactly when unit_values
    exceeds target / payment. Sorting once makes this a lookup instead of a pass over every
    path for every payment. A payment of 0 never reaches a positive target.
    """
    payments = np.asarray(payments, dtype=float)
    ordered = np.sort(unit_values)
    n = len(ordered)
    probs = np.zeros(len(payments))
    positive = payments > 0
    thresholds = target / payments[positive]
    # side="right": count of values <= threshold, so n - that is the count strictly above.
    probs[positive] = (n - np.searchsorted(ordered, thresholds, side="right")) / n
    return probs


def required_contribution(
    payments: Sequence[float], probabilities: Sequence[float], target: float = 0.95
) -> Optional[float]:
    """Smallest payment tested whose probability reaches `target`, or None if none does."""
    for payment, probability in zip(payments, probabilities):
        if probability >= target:
            return float(payment)
    return None
