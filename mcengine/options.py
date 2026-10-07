"""Monte Carlo pricing of European, Asian and barrier options on one underlying asset.

Every price is simulated with mcengine.gbm using the risk-neutral drift r - q. Black-Scholes
(mcengine.black_scholes) is the check for the European price and the Greeks, and the
closed-form geometric Asian price is both the check for that payoff and the control variate
for the arithmetic Asian option.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import black_scholes as bs
from .gbm import simulate_gbm_paths

OPTION_TYPES = ("call", "put")
BARRIER_TYPES = ("up-and-out", "down-and-out", "up-and-in", "down-and-in")
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MarketData:
    """The market an option is priced in."""

    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float

    def __post_init__(self) -> None:
        if self.spot_price <= 0:
            raise ValueError("Spot price must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")


@dataclass(frozen=True)
class OptionContract:
    """An option. The barrier fields are only used by price_barrier."""

    option_type: str
    strike: float
    expiry: float
    barrier_level: Optional[float] = None
    barrier_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.option_type not in OPTION_TYPES:
            raise ValueError(f"option_type must be 'call' or 'put', not {self.option_type!r}.")
        if self.strike <= 0:
            raise ValueError("Strike must be positive.")
        if self.expiry <= 0:
            raise ValueError("Expiry must be positive.")
        if self.barrier_type is not None and self.barrier_type not in BARRIER_TYPES:
            raise ValueError(f"Unknown barrier type: {self.barrier_type!r}.")


@dataclass
class PricingResult:
    """A price with its standard error and 95% confidence interval, and Greeks when computed."""

    price: float
    std_error: float
    confidence_interval: Tuple[float, float]
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    num_paths: int = 0
    num_steps: int = 0


def _payoff(option_type: str, underlying: np.ndarray, strike: float) -> np.ndarray:
    if option_type == "call":
        return np.maximum(underlying - strike, 0.0)
    return np.maximum(strike - underlying, 0.0)


def _pair_average(values: np.ndarray) -> np.ndarray:
    """Average each path with its antithetic partner (row i with row i + n/2)."""
    half = len(values) // 2
    return 0.5 * (values[:half] + values[half:])


def _mean_and_error(values: np.ndarray, antithetic: bool) -> Tuple[float, float]:
    """Sample mean and its standard error.

    The two paths of an antithetic pair are correlated, so they are averaged first and the
    standard error is computed from the independent pair averages. Treating all paths as
    independent would understate the error.
    """
    if antithetic:
        values = _pair_average(values)
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def _control_variate_estimate(
    values: np.ndarray, control: np.ndarray, control_mean: float, antithetic: bool
) -> Tuple[float, float]:
    """Mean and standard error of values after subtracting beta * (control - its known mean).

    beta = cov(values, control) / var(control) is the slope that removes the most variance.
    It is estimated from the same sample, which adds a bias of order 1 / n_paths that is
    negligible next to the standard error.
    """
    if antithetic:
        values, control = _pair_average(values), _pair_average(control)
    var = control.var(ddof=1)
    beta = 0.0 if var == 0 else float(np.cov(values, control, ddof=1)[0, 1] / var)
    adjusted = values - beta * (control - control_mean)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / math.sqrt(len(adjusted)))


def _result(
    mean: float, error: float, rate: float, expiry: float, n_paths: int, n_steps: int, **greeks: float
) -> PricingResult:
    discount = math.exp(-rate * expiry)
    price, std_error = discount * mean, discount * error
    return PricingResult(
        price=price,
        std_error=std_error,
        confidence_interval=(price - Z_95 * std_error, price + Z_95 * std_error),
        num_paths=n_paths,
        num_steps=n_steps,
        **greeks,
    )


class MonteCarloEngine:
    """Prices options by simulation. Give a seed to get the same prices every run."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def _draw_seed(self) -> int:
        return int(self._rng.integers(2**62))

    @staticmethod
    def _paths(
        market: MarketData, expiry: float, n_paths: int, n_steps: int, antithetic: bool, seed: int
    ) -> np.ndarray:
        return simulate_gbm_paths(
            market.spot_price,
            market.risk_free_rate - market.dividend_yield,
            market.volatility,
            expiry,
            n_steps,
            n_paths,
            antithetic=antithetic,
            rng=seed,
        )

    def price_european(
        self,
        market: MarketData,
        option: OptionContract,
        num_paths: int = 100_000,
        num_steps: int = 1,
        antithetic: bool = True,
        control_variate: bool = True,
        compute_greeks: bool = True,
    ) -> PricingResult:
        """Price a European option.

        Only the final price matters for the payoff and the simulation is exact, so one step
        gives the same distribution as 252. num_steps is kept so that the same paths can be
        used to look at the path, not because the price needs it.

        The control variate is the final price S(T), whose mean S0 * exp((r - q) * T) is
        known exactly. A call payoff moves with S(T), so subtracting the part of the payoff
        explained by S(T) removes most of the noise.

        The Greeks are finite differences of the Monte Carlo price. Every bumped price reuses
        the same random numbers as the base price (common random numbers). With fresh random
        numbers for each bump, the noise in each price is larger than the change being
        measured and the Greeks are meaningless.
        """
        seed = self._draw_seed()
        paths = self._paths(market, option.expiry, num_paths, num_steps, antithetic, seed)
        final = paths[:, -1]
        payoffs = _payoff(option.option_type, final, option.strike)

        if control_variate:
            forward = market.spot_price * math.exp(
                (market.risk_free_rate - market.dividend_yield) * option.expiry
            )
            mean, error = _control_variate_estimate(payoffs, final, forward, antithetic)
        else:
            mean, error = _mean_and_error(payoffs, antithetic)

        greeks = {}
        if compute_greeks:
            greeks = self._greeks(market, option, num_paths, num_steps, antithetic, seed)
        return _result(mean, error, market.risk_free_rate, option.expiry, num_paths, num_steps, **greeks)

    def _plain_price(
        self, market: MarketData, option: OptionContract, n_paths: int, n_steps: int,
        antithetic: bool, seed: int,
    ) -> float:
        paths = self._paths(market, option.expiry, n_paths, n_steps, antithetic, seed)
        payoffs = _payoff(option.option_type, paths[:, -1], option.strike)
        return math.exp(-market.risk_free_rate * option.expiry) * float(payoffs.mean())

    def _greeks(
        self, market: MarketData, option: OptionContract, n_paths: int, n_steps: int,
        antithetic: bool, seed: int,
    ) -> dict:
        def at(spot: float, vol: float) -> float:
            bumped = MarketData(spot, market.risk_free_rate, market.dividend_yield, vol)
            return self._plain_price(bumped, option, n_paths, n_steps, antithetic, seed)

        spot, vol = market.spot_price, market.volatility
        h = 0.02 * spot
        dv = min(0.01, vol / 2)
        mid = at(spot, vol)
        up, down = at(spot + h, vol), at(spot - h, vol)
        return {
            "delta": (up - down) / (2 * h),
            "gamma": (up - 2 * mid + down) / h**2,
            "vega": (at(spot, vol + dv) - at(spot, vol - dv)) / (2 * dv),
        }

    def price_asian(
        self,
        market: MarketData,
        option: OptionContract,
        num_paths: int = 100_000,
        num_steps: int = 52,
        antithetic: bool = False,
        control_variate: bool = True,
    ) -> PricingResult:
        """Price an option on the arithmetic average of num_steps equally spaced prices.

        The average uses the prices at times T/n, 2T/n, ..., T. The starting price is not
        included. The control variate is the same option on the geometric average, which has
        a closed form (black_scholes.geometric_asian_price) and moves almost exactly with
        the arithmetic one.
        """
        seed = self._draw_seed()
        paths = self._paths(market, option.expiry, num_paths, num_steps, antithetic, seed)
        fixings = paths[:, 1:]
        payoffs = _payoff(option.option_type, fixings.mean(axis=1), option.strike)

        if control_variate:
            geometric = _payoff(option.option_type, np.exp(np.log(fixings).mean(axis=1)), option.strike)
            exact = bs.geometric_asian_price(
                option.option_type, market.spot_price, option.strike, option.expiry,
                market.risk_free_rate, market.dividend_yield, market.volatility, num_steps,
            )
            geometric_mean = exact * math.exp(market.risk_free_rate * option.expiry)
            mean, error = _control_variate_estimate(payoffs, geometric, geometric_mean, antithetic)
        else:
            mean, error = _mean_and_error(payoffs, antithetic)
        return _result(mean, error, market.risk_free_rate, option.expiry, num_paths, num_steps)

    def price_barrier(
        self,
        market: MarketData,
        option: OptionContract,
        num_paths: int = 100_000,
        num_steps: int = 252,
        antithetic: bool = False,
    ) -> PricingResult:
        """Price a barrier option.

        'Out' options pay the vanilla payoff unless the barrier is touched; 'in' options pay
        it only if the barrier is touched. The barrier is checked at the num_steps simulated
        times only, so a crossing between two times is missed. That makes knock-outs worth
        slightly more, and knock-ins slightly less, than the same option monitored
        continuously.
        """
        if option.barrier_level is None or option.barrier_type is None:
            raise ValueError("A barrier option needs barrier_level and barrier_type.")
        going_up = option.barrier_type.startswith("up")
        if going_up and option.barrier_level <= market.spot_price:
            raise ValueError("An up barrier must be above the spot price.")
        if not going_up and option.barrier_level >= market.spot_price:
            raise ValueError("A down barrier must be below the spot price.")

        seed = self._draw_seed()
        paths = self._paths(market, option.expiry, num_paths, num_steps, antithetic, seed)
        if going_up:
            touched = (paths[:, 1:] >= option.barrier_level).any(axis=1)
        else:
            touched = (paths[:, 1:] <= option.barrier_level).any(axis=1)

        payoffs = _payoff(option.option_type, paths[:, -1], option.strike)
        alive = touched if option.barrier_type.endswith("in") else ~touched
        mean, error = _mean_and_error(payoffs * alive, antithetic)
        return _result(mean, error, market.risk_free_rate, option.expiry, num_paths, num_steps)
