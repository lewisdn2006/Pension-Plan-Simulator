"""Price a European, an Asian and a barrier option and calibrate an implied volatility.

Run:  python examples/price_options.py [--paths 100000]
Plots are saved in examples/output/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from mcengine import black_scholes as bs  # noqa: E402
from mcengine.gbm import simulate_gbm_paths  # noqa: E402
from mcengine.implied_vol import solve_iv  # noqa: E402
from mcengine.options import MarketData, MonteCarloEngine, OptionContract  # noqa: E402
from mcengine.plotting import plot_convergence, plot_paths  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "output"


def show(title, result):
    low, high = result.confidence_interval
    print(f"{title}: {result.price:.4f}  (standard error {result.std_error:.4f}, 95% CI {low:.4f} to {high:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=100_000, help="paths per price")
    args = parser.parse_args()

    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02, volatility=0.25)
    engine = MonteCarloEngine(seed=42)

    call = OptionContract("call", strike=105.0, expiry=1.0)
    result = engine.price_european(market, call, num_paths=args.paths)
    exact = bs.call_price(100.0, 105.0, 1.0, 0.05, 0.02, 0.25)
    print("European call, strike 105, one year")
    show("  Monte Carlo ", result)
    print(f"  Black-Scholes: {exact:.4f}   error {result.price - exact:+.4f}")
    print(f"  Delta {result.delta:.4f} (exact {bs.delta('call', 100, 105, 1, 0.05, 0.02, 0.25):.4f}), "
          f"gamma {result.gamma:.5f} (exact {bs.gamma(100, 105, 1, 0.05, 0.02, 0.25):.5f}), "
          f"vega {result.vega:.2f} (exact {bs.vega(100, 105, 1, 0.05, 0.02, 0.25):.2f})")

    sample = simulate_gbm_paths(100.0, 0.03, 0.25, 1.0, 252, 2000, rng=1)
    plot_paths(sample, 1.0, OUTPUT / "option_paths.png", "Simulated price paths (risk-neutral drift)", "Price")

    counts = [10**k for k in range(2, 7)]
    prices = [MonteCarloEngine(seed=k).price_european(market, call, n, compute_greeks=False).price
              for k, n in enumerate(counts)]
    plot_convergence(counts, prices, exact, OUTPUT / "convergence.png")

    print("\nAsian call (average of 52 weekly prices), strike 100")
    show("  Monte Carlo ", engine.price_asian(market, OptionContract("call", 100.0, 1.0), args.paths))

    print("\nUp-and-out call, strike 100, barrier 120, checked 252 times")
    show("  Monte Carlo ", engine.price_barrier(
        market, OptionContract("call", 100.0, 1.0, 120.0, "up-and-out"), args.paths))

    print("\nImplied volatility of the 105 call if it traded at 8.50")
    vol = solve_iv(8.50, 100.0, 105.0, 1.0, 0.05, 0.02, "call")
    check = bs.call_price(100.0, 105.0, 1.0, 0.05, 0.02, vol)
    print(f"  {vol:.2%}; Black-Scholes at that volatility gives {check:.4f}")
    np.set_printoptions(suppress=True)


if __name__ == "__main__":
    main()
