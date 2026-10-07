"""Closed-form prices and Greeks, used to check the Monte Carlo prices and for implied volatility.

All functions take the spot price s, strike k, time to expiry t in years, risk-free rate r,
continuous dividend yield q and volatility sigma. Vega is the change in price for a change of
1.00 in volatility (so divide by 100 for "per volatility point"); theta is per year.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _check(s: float, k: float, t: float, sigma: float) -> None:
    if s <= 0 or k <= 0:
        raise ValueError("Spot and strike must be positive.")
    if t <= 0:
        raise ValueError("Time to expiry must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility must be positive.")


def d1(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    _check(s, k, t, sigma)
    return (math.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))


def d2(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    return d1(s, k, t, r, q, sigma) - sigma * math.sqrt(t)


def call_price(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    b = a - sigma * math.sqrt(t)
    return s * math.exp(-q * t) * norm.cdf(a) - k * math.exp(-r * t) * norm.cdf(b)


def put_price(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    b = a - sigma * math.sqrt(t)
    return k * math.exp(-r * t) * norm.cdf(-b) - s * math.exp(-q * t) * norm.cdf(-a)


def price(option_type: str, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    """Price of a European 'call' or 'put'."""
    if option_type == "call":
        return call_price(s, k, t, r, q, sigma)
    if option_type == "put":
        return put_price(s, k, t, r, q, sigma)
    raise ValueError(f"option_type must be 'call' or 'put', not {option_type!r}.")


def delta(option_type: str, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    if option_type == "call":
        return math.exp(-q * t) * norm.cdf(a)
    if option_type == "put":
        return -math.exp(-q * t) * norm.cdf(-a)
    raise ValueError(f"option_type must be 'call' or 'put', not {option_type!r}.")


def gamma(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    return math.exp(-q * t) * norm.pdf(a) / (s * sigma * math.sqrt(t))


def vega(s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    return s * math.exp(-q * t) * norm.pdf(a) * math.sqrt(t)


def theta(option_type: str, s: float, k: float, t: float, r: float, q: float, sigma: float) -> float:
    a = d1(s, k, t, r, q, sigma)
    b = a - sigma * math.sqrt(t)
    decay = -s * math.exp(-q * t) * norm.pdf(a) * sigma / (2 * math.sqrt(t))
    if option_type == "call":
        return decay + q * s * math.exp(-q * t) * norm.cdf(a) - r * k * math.exp(-r * t) * norm.cdf(b)
    if option_type == "put":
        return decay - q * s * math.exp(-q * t) * norm.cdf(-a) + r * k * math.exp(-r * t) * norm.cdf(-b)
    raise ValueError(f"option_type must be 'call' or 'put', not {option_type!r}.")


def geometric_asian_price(
    option_type: str, s: float, k: float, t: float, r: float, q: float, sigma: float, n_fixings: int
) -> float:
    """Price of an option on the geometric average of n_fixings equally spaced prices.

    The fixings are at t/n, 2t/n, ..., t. The geometric average of lognormal prices is itself
    lognormal, so there is a closed form. It is used as the control variate for the arithmetic
    Asian option, whose price has no closed form.
    """
    _check(s, k, t, sigma)
    n = n_fixings
    mean_log = math.log(s) + (r - q - 0.5 * sigma**2) * t * (n + 1) / (2 * n)
    var_log = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    sd = math.sqrt(var_log)
    forward = math.exp(mean_log + 0.5 * var_log)
    a = (mean_log + var_log - math.log(k)) / sd
    b = a - sd
    discount = math.exp(-r * t)
    if option_type == "call":
        return discount * (forward * norm.cdf(a) - k * norm.cdf(b))
    if option_type == "put":
        return discount * (k * norm.cdf(-b) - forward * norm.cdf(-a))
    raise ValueError(f"option_type must be 'call' or 'put', not {option_type!r}.")
