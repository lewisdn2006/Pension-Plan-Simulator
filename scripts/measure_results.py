"""Measure every number quoted in the README and write results/measured_results.md
(results/measured_results_full.md with --full).

Run:  python scripts/measure_results.py            (quick: 10,000 pension paths)
      python scripts/measure_results.py --full     (100,000 pension paths, as in the original model)

All random numbers are seeded, so the same command gives the same file.
"""

import argparse
import math
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcengine import black_scholes as bs  # noqa: E402
from mcengine import pension  # noqa: E402
from mcengine.gbm import simulate_gbm_paths  # noqa: E402
from mcengine.implied_vol import solve_iv  # noqa: E402
from mcengine.options import MarketData, MonteCarloEngine, OptionContract  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
MARKET = MarketData(100.0, 0.05, 0.02, 0.25)
CALL = OptionContract("call", 105.0, 1.0)
EXACT = bs.call_price(100.0, 105.0, 1.0, 0.05, 0.02, 0.25)
lines = []


def out(text=""):
    lines.append(text)
    print(text)


def table(header, rows):
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out("| " + " | ".join(str(c) for c in row) + " |")
    out()


def european_modes(runs, paths):
    out("## 1. European call: accuracy, variance reduction and the reported error")
    out()
    out(f"European call, S = 100, K = 105, T = 1 year, r = 5%, q = 2%, volatility 25%. "
        f"Black-Scholes price {EXACT:.4f}. Each row is {runs} independent runs of {paths:,} paths "
        f"(seeds 0 to {runs - 1}). The simulation is exact, so one time step per path.")
    out()
    rows = []
    modes = [("plain", False, False), ("antithetic only", True, False),
             ("control variate only", False, True), ("antithetic + control variate", True, True)]
    base_sd = None
    for name, anti, cv in modes:
        prices, errors, hits = [], [], 0
        for seed in range(runs):
            r = MonteCarloEngine(seed=seed).price_european(
                MARKET, CALL, paths, antithetic=anti, control_variate=cv, compute_greeks=False)
            prices.append(r.price)
            errors.append(r.std_error)
            hits += r.confidence_interval[0] <= EXACT <= r.confidence_interval[1]
        prices = np.array(prices)
        sd = prices.std(ddof=1)
        base_sd = base_sd or sd
        rows.append([name, f"{prices.mean():.4f}", f"{prices.mean() - EXACT:+.4f}", f"{sd:.4f}",
                     f"{base_sd / sd:.2f}x", f"{np.mean(errors):.4f}", f"{np.mean(errors) / sd:.2f}",
                     f"{100 * hits / runs:.1f}%"])
    table(["mode", "mean price", "mean - exact", "SD across runs", "SD smaller by",
           "mean reported SE", "reported SE / SD", "95% CI covers exact"], rows)
    return rows


def convergence():
    out("## 2. Convergence")
    out()
    out("Root-mean-square error against Black-Scholes over 50 runs at each size (control variate on).")
    out()
    rows, rms_list, sizes = [], [], [10**3, 10**4, 10**5, 10**6]
    for n in sizes:
        errs = [MonteCarloEngine(seed=s).price_european(MARKET, CALL, n, compute_greeks=False).price - EXACT
                for s in range(50)]
        rms_list.append(math.sqrt(np.mean(np.square(errs))))
        rows.append([f"{n:,}", f"{rms_list[-1]:.5f}"])
    table(["paths", "RMS error"], rows)
    slope = np.polyfit(np.log(sizes), np.log(rms_list), 1)[0]
    out(f"Fitted slope of log(error) against log(paths): {slope:.2f} (theory: -0.50).")
    out()


def greeks():
    out("## 3. Greeks (finite differences with common random numbers)")
    out()
    n = 400_000
    r = MonteCarloEngine(seed=8).price_european(MARKET, CALL, n)
    args = (100.0, 105.0, 1.0, 0.05, 0.02, 0.25)
    rows = [["delta", f"{r.delta:.4f}", f"{bs.delta('call', *args):.4f}"],
            ["gamma", f"{r.gamma:.5f}", f"{bs.gamma(*args):.5f}"],
            ["vega (per 1.00 of volatility)", f"{r.vega:.2f}", f"{bs.vega(*args):.2f}"]]
    out(f"{n:,} paths, seed 8.")
    out()
    table(["Greek", "Monte Carlo", "Black-Scholes"], rows)


def asian(paths):
    out("## 4. Asian call (average of 52 weekly prices)")
    out()
    option = OptionContract("call", 100.0, 1.0)
    geo_exact = bs.geometric_asian_price("call", 100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 52)
    sim = simulate_gbm_paths(100.0, 0.05 - 0.02, 0.25, 1.0, 52, paths, rng=1)
    geo_payoff = np.maximum(np.exp(np.log(sim[:, 1:]).mean(axis=1)) - 100.0, 0.0)
    geo_mc = math.exp(-0.05) * geo_payoff.mean()
    geo_se = math.exp(-0.05) * geo_payoff.std(ddof=1) / math.sqrt(paths)
    plain = MonteCarloEngine(seed=3).price_asian(MARKET, option, paths, control_variate=False)
    cv = MonteCarloEngine(seed=4).price_asian(MARKET, option, paths, control_variate=True)
    out(f"{paths:,} paths.")
    out()
    table(["quantity", "price", "standard error"],
          [["geometric Asian, closed form", f"{geo_exact:.4f}", "-"],
           ["geometric Asian, simulated", f"{geo_mc:.4f}", f"{geo_se:.4f}"],
           ["arithmetic Asian, plain", f"{plain.price:.4f}", f"{plain.std_error:.4f}"],
           ["arithmetic Asian, geometric control variate", f"{cv.price:.4f}", f"{cv.std_error:.5f}"]])
    out(f"The control variate makes the standard error {plain.std_error / cv.std_error:.0f} times smaller.")
    out()


def barrier(paths):
    out("## 5. Barrier options (up barrier 120, call strike 100)")
    out()
    vanilla = bs.call_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25)
    out(f"{paths:,} paths. Vanilla call (Black-Scholes): {vanilla:.4f}.")
    out()
    rows = []
    for steps in (12, 52, 252, 1000):
        o = MonteCarloEngine(seed=5).price_barrier(
            MARKET, OptionContract("call", 100.0, 1.0, 120.0, "up-and-out"), paths, steps)
        i = MonteCarloEngine(seed=5).price_barrier(
            MARKET, OptionContract("call", 100.0, 1.0, 120.0, "up-and-in"), paths, steps)
        rows.append([steps, f"{o.price:.4f}", f"{i.price:.4f}", f"{o.price + i.price:.4f}"])
    table(["barrier checks", "up-and-out", "up-and-in", "sum"], rows)
    out("The sum is the vanilla price on the same paths. The knock-out falls and the knock-in "
        "rises as the barrier is checked more often, because fewer crossings are missed.")
    out()


def implied_vol():
    out("## 6. Implied volatility round trip")
    out()
    worst, count = 0.0, 0
    for kind in ("call", "put"):
        for sigma in np.arange(0.05, 1.51, 0.05):
            for strike in range(70, 141, 10):
                if bs.vega(100, strike, 0.75, 0.03, 0.01, sigma) < 1e-3:
                    continue
                price = bs.price(kind, 100, strike, 0.75, 0.03, 0.01, sigma)
                worst = max(worst, abs(solve_iv(price, 100, strike, 0.75, 0.03, 0.01, kind) - sigma))
                count += 1
    out(f"{count} options (calls and puts, strikes 70 to 140, volatilities 5% to 150%): "
        f"largest error in the recovered volatility {worst:.1e}.")
    out()


def pension_section(n_paths):
    out("## 7. Pension model")
    out()
    note = "" if n_paths >= 100_000 else " (the original model used 100,000; run with --full to match)"
    out(f"{n_paths:,} simulated index paths per case{note}.")
    out()
    start = time.time()
    paths = pension.simulate_index_paths(0.05, n_paths, 40, rng=0)
    values = pension.fund_values(paths, 40, 1000.0)
    loss, double, high = pension.outcome_probabilities(values, 40, 1000.0)
    out("**40 years, 5% growth, 1,000 a month (480,000 paid in)**")
    out()
    table(["outcome", "probability"],
          [["finish with less than was paid in", f"{loss:.4f}"],
           ["finish with more than double what was paid in", f"{double:.4f}"],
           ["finish with more than 2 million", f"{high:.4f}"]])
    target, growth = 1_000_000.0, [0.03, 0.05, 0.07]
    needed = {}
    for years, top in [(40, 4000.0), (20, 10000.0)]:
        payments = np.arange(0, top, 20.0)
        for mu in growth:
            unit = pension.unit_fund_values(pension.simulate_index_paths(mu, n_paths, years, rng=years), years)
            needed[years, mu] = pension.required_contribution(
                payments, pension.probability_curve(unit, payments, target))
    elapsed = time.time() - start
    out("**Monthly payment (in steps of 20) for a 95% chance of at least 1 million**")
    out()
    rows = []
    for mu in growth:
        a, b = needed[40, mu], needed[20, mu]
        rows.append([f"{mu:.0%}", f"{a:,.0f}" if a is not None else "none up to 4,000",
                     f"{b:,.0f}" if b is not None else "none up to 10,000",
                     f"{b - a:,.0f}" if a is not None and b is not None else "-"])
    table(["growth rate", "40 years", "20 years", "extra for starting 20 years later"], rows)
    out(f"All of the above took {elapsed:.1f} seconds.")
    out()

    out("**Checking the shortcut and timing it.** The original code recomputed the fund on every "
        "path for every payment amount. This package computes the fund for a payment of 1 once and "
        "scales it. Both give the same probabilities (compared below on 2,000 paths and 200 payments).")
    out()
    small = pension.simulate_index_paths(0.05, 2000, 40, rng=1)
    payments = np.arange(0, 4000.0, 20.0)
    t0 = time.time()
    naive = np.array([np.mean(pension.fund_values(small, 40, m) > target) for m in payments])
    t_naive = time.time() - t0
    t0 = time.time()
    fast = pension.probability_curve(pension.unit_fund_values(small, 40), payments, target)
    t_fast = time.time() - t0
    out(f"Largest difference between the two probability curves: {np.abs(naive - fast).max():.1e}. "
        f"Payment-by-payment: {t_naive:.3f} s. Scaled: {t_fast:.4f} s ({t_naive / t_fast:.0f} times faster).")
    out()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="use 100,000 pension paths")
    args = parser.parse_args()
    started = time.time()
    out("# Measured results")
    out()
    out(f"Produced by `scripts/measure_results.py{' --full' if args.full else ''}` "
        f"with Python {platform.python_version()}, numpy {np.__version__}. All random numbers are seeded.")
    out()
    european_modes(runs=400, paths=20_000)
    convergence()
    greeks()
    asian(200_000)
    barrier(200_000)
    implied_vol()
    pension_section(100_000 if args.full else 10_000)
    out(f"Total run time: {time.time() - started:.0f} seconds.")
    RESULTS.mkdir(exist_ok=True)
    name = "measured_results_full.md" if args.full else "measured_results.md"
    (RESULTS / name).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
