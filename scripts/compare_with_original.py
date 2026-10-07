"""Run the two original scripts and this package side by side.

The original scripts (option_pricer.py and pension_simulator.py) are not part of this
repository. Point --original-dir at a folder that contains them (subfolders are searched):

    python scripts/compare_with_original.py --original-dir path/to/folder

It measures, on the original code, the faults that this package corrects, and checks that the
pension model gives the same numbers as before for the same seed.
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from mcengine import black_scholes as bs  # noqa: E402
from mcengine import pension  # noqa: E402
from mcengine.gbm import simulate_gbm_paths  # noqa: E402
from mcengine.options import MarketData, MonteCarloEngine, OptionContract  # noqa: E402


def find(folder: Path, name: str) -> Path:
    matches = sorted(folder.rglob(name))
    if not matches:
        raise SystemExit(f"{name} not found under {folder}")
    return matches[0]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=200, help="repeated runs for the error check")
    args = parser.parse_args()

    op = load(find(args.original_dir, "option_pricer.py"), "original_option_pricer")
    pn = load(find(args.original_dir, "pension_simulator.py"), "original_pension")

    market = op.MarketData(100.0, 0.05, 0.02, 0.25)
    new_market = MarketData(100.0, 0.05, 0.02, 0.25)
    exact = bs.call_price(100.0, 105.0, 1.0, 0.05, 0.02, 0.25)

    print("1. Pension paths and funds, original against this package, same seed")
    old_paths = pn.simulate_gbm(0.05, 2000, 40, 480, seed=7)
    new_paths = simulate_gbm_paths(pn.S0, 0.05, pn.sigma, 40, 480, 2000, rng=7)
    old_values = pn.pension_value(old_paths, 40, 1000)
    new_values = pension.fund_values(new_paths, 40, 1000.0)
    print(f"   largest difference in a path: {np.abs(old_paths - new_paths).max():.1e}")
    print(f"   largest difference in a final fund: {np.abs(old_values - new_values).max():.1e}")

    print("\n2. Knock-in against knock-out, original code (barrier 120, same seed)")
    out_opt = op.OptionContract("call", 100.0, 1.0, "barrier", 120.0, "up-and-out")
    in_opt = op.OptionContract("call", 100.0, 1.0, "barrier", 120.0, "up-and-in")
    old_out = op.MonteCarloEngine(seed=5).price_barrier(market, out_opt, 100_000).price
    old_in = op.MonteCarloEngine(seed=5).price_barrier(market, in_opt, 100_000).price
    new_out = MonteCarloEngine(seed=5).price_barrier(
        new_market, OptionContract("call", 100.0, 1.0, 120.0, "up-and-out"), 100_000).price
    new_in = MonteCarloEngine(seed=5).price_barrier(
        new_market, OptionContract("call", 100.0, 1.0, 120.0, "up-and-in"), 100_000).price
    print(f"   original: up-and-out {old_out:.4f}, up-and-in {old_in:.4f}")
    print(f"   this package: up-and-out {new_out:.4f}, up-and-in {new_in:.4f}")

    print(f"\n3. Reported error and coverage, original default settings ({args.runs} runs of 20,000 paths x 252 steps)")
    call = op.OptionContract("call", 105.0, 1.0)
    prices, errors, hits = [], [], 0
    for seed in range(args.runs):
        r = op.MonteCarloEngine(seed=seed).price_european(market, call, 20_000, 252, compute_greeks=False)
        prices.append(r.price)
        errors.append(r.std_error)
        hits += r.confidence_interval[0] <= exact <= r.confidence_interval[1]
    sd = np.std(prices, ddof=1)
    print(f"   SD of the price across runs {sd:.4f}, mean reported standard error {np.mean(errors):.4f} "
          f"(ratio {np.mean(errors) / sd:.2f}), 95% interval covers the exact price {100 * hits / args.runs:.1f}% of the time")

    print("\n4. Greeks, original code, 100,000 paths, five seeds (exact: delta %.4f, gamma %.5f, vega %.2f per 1.00, %.4f per 0.01)" % (
        bs.delta('call', 100, 105, 1, .05, .02, .25), bs.gamma(100, 105, 1, .05, .02, .25),
        bs.vega(100, 105, 1, .05, .02, .25), bs.vega(100, 105, 1, .05, .02, .25) / 100))
    for seed in range(5):
        r = op.MonteCarloEngine(seed=seed).price_european(market, call, 100_000, 252)
        print(f"   seed {seed}: delta {r.delta:.4f}, gamma {r.gamma:.5f}, vega {r.vega:.2f}")
    print("   this package (common random numbers):")
    for seed in range(5):
        r = MonteCarloEngine(seed=seed).price_european(new_market, OptionContract('call', 105.0, 1.0), 100_000)
        print(f"   seed {seed}: delta {r.delta:.4f}, gamma {r.gamma:.5f}, vega {r.vega:.2f}")

    print("\n5. Implied volatility of a price below the lowest the model can give (call, S=150, K=100, price 1.00)")
    print(f"   original returns {op.solve_iv(1.0, 150.0, 100.0, 1.0, 0.05, 0.0, 'call'):.4f}")
    try:
        from mcengine.implied_vol import solve_iv
        solve_iv(1.0, 150.0, 100.0, 1.0, 0.05, 0.0, "call")
    except ValueError as error:
        print(f"   this package raises: {error}")

    print("\n6. Probability curve over 200 payments, 2,000 paths, 40 years")
    mu_list, payments = [0.05], np.arange(0, 4000, 20)
    start = time.time()
    _, old_matrix = pn.probability_curves(40, 1_000_000, mu_list, 20, 4000, 2000)
    t_old = time.time() - start
    paths = pension.simulate_index_paths(0.05, 2000, 40)
    start = time.time()
    pension.probability_curve(pension.unit_fund_values(paths, 40), payments, 1_000_000.0)
    t_new = time.time() - start
    print(f"   original {t_old:.2f} s, this package {t_new:.4f} s ({t_old / t_new:.0f} times faster)")


if __name__ == "__main__":
    main()
