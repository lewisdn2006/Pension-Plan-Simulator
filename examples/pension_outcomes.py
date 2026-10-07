"""Pension outcomes: one scenario in detail, then the monthly payment needed for a 95% chance of 1 million.

Run:  python examples/pension_outcomes.py [--paths 10000]
The original model used 100,000 paths; 10,000 is the quick default. Plots go in examples/output/.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcengine import pension  # noqa: E402
from mcengine.plotting import plot_histogram, plot_paths, plot_probability_curves  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=10_000, help="simulated index paths")
    args = parser.parse_args()
    start = time.time()

    # One scenario in detail: 40 years, 5% growth, 1,000 a month.
    paths = pension.simulate_index_paths(0.05, args.paths, 40, rng=0)
    values = pension.fund_values(paths, 40, 1000.0)
    plot_paths(paths, 40, OUTPUT / "pension_paths.png", "Simulated share index, 5% expected growth", "Index")
    plot_histogram(values, OUTPUT / "pension_final_values.png", "Fund after 40 years of 1,000 a month", "Final fund")
    loss, double, high = pension.outcome_probabilities(values, 40, 1000.0)
    counts, edges = np.histogram(values, bins=np.arange(0, values.max() + 50_000, 50_000))
    print(f"40 years, 5% growth, 1,000 a month, {args.paths:,} paths")
    print(f"  chance of finishing with less than was paid in: {loss:.4f}")
    print(f"  chance of more than double what was paid in:    {double:.4f}")
    print(f"  chance of more than 2 million:                  {high:.4f}")
    print(f"  fullest 50,000-wide bin starts at {edges[counts.argmax()]:,.0f} and holds {counts.max():,} paths")

    # Monthly payment for a 95% chance of at least 1 million.
    target = 1_000_000.0
    growth_rates = [0.03, 0.05, 0.07]
    needed = {}
    for years, top in [(40, 4000.0), (20, 10000.0)]:
        payments = np.arange(0, top, 20.0)
        curves = {}
        for mu in growth_rates:
            unit = pension.unit_fund_values(pension.simulate_index_paths(mu, args.paths, years, rng=years), years)
            probs = pension.probability_curve(unit, payments, target)
            curves[f"mu = {mu}"] = probs
            needed[years, mu] = pension.required_contribution(payments, probs)
        plot_probability_curves(
            payments, curves, target, OUTPUT / f"pension_probability_curves_{years}_years.png",
            f"Chance of a 1 million fund after {years} years",
        )

    print("\nMonthly payment for a 95% chance of at least 1 million")
    for mu in growth_rates:
        a, b = needed[40, mu], needed[20, mu]
        gap = "" if a is None or b is None else f"  (starting 20 years later costs {b - a:,.0f} more)"
        print(f"  mu = {mu}:  {a if a is not None else 'more than 4,000'} over 40 years, "
              f"{b if b is not None else 'more than 10,000'} over 20 years{gap}")
    print(f"\nTotal time: {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
