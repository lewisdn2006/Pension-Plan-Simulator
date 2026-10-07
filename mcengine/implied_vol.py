"""Implied volatility: the volatility that makes the Black-Scholes price equal an observed price."""

from __future__ import annotations

import math

import pandas as pd
from scipy.optimize import brentq

from . import black_scholes as bs

VOL_LOW = 1e-4
VOL_HIGH = 5.0


def solve_iv(
    market_price: float, s: float, k: float, t: float, r: float, q: float, option_type: str
) -> float:
    """Implied volatility of a European option, found with Brent's method.

    The Black-Scholes price rises with volatility, so there is at most one answer. If the
    observed price is outside the range the model can produce for volatilities between
    VOL_LOW and VOL_HIGH (for example below the discounted intrinsic value), no volatility
    fits and a ValueError is raised instead of returning a meaningless number.
    """

    def gap(sigma: float) -> float:
        return bs.price(option_type, s, k, t, r, q, sigma) - market_price

    if gap(VOL_LOW) > 0 or gap(VOL_HIGH) < 0:
        low = bs.price(option_type, s, k, t, r, q, VOL_LOW)
        high = bs.price(option_type, s, k, t, r, q, VOL_HIGH)
        raise ValueError(
            f"Price {market_price:.4f} is outside the range {low:.4f} to {high:.4f} "
            "that volatilities between 0.01% and 500% can give."
        )
    return brentq(gap, VOL_LOW, VOL_HIGH, xtol=1e-12)


def build_vol_surface(quotes: pd.DataFrame, s: float, r: float, q: float) -> pd.DataFrame:
    """Add an 'implied_vol' column to a table of option quotes.

    Args:
        quotes: a table with columns 'strike', 'expiry', 'price' and 'type' ('call' or 'put').

    Returns:
        A copy of the table with an extra column. Quotes that cannot be inverted get NaN.
    """
    out = quotes.copy()
    values = []
    for _, row in out.iterrows():
        try:
            values.append(solve_iv(row["price"], s, row["strike"], row["expiry"], r, q, row["type"]))
        except ValueError:
            values.append(math.nan)
    out["implied_vol"] = values
    return out
