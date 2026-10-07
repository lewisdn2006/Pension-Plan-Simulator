import math

import numpy as np
import pytest

from mcengine import black_scholes as bs
from mcengine.gbm import simulate_gbm_paths

ARGS = (100.0, 105.0, 1.0, 0.05, 0.02, 0.25)


def test_textbook_values():
    # S = K = 100, T = 1, r = 5%, no dividends, vol 20%.
    assert bs.call_price(100, 100, 1, 0.05, 0.0, 0.2) == pytest.approx(10.4506, abs=1e-4)
    assert bs.put_price(100, 100, 1, 0.05, 0.0, 0.2) == pytest.approx(5.5735, abs=1e-4)


def test_value_used_throughout_the_project():
    assert bs.call_price(*ARGS) == pytest.approx(8.9412, abs=1e-4)


def test_put_call_parity():
    s, k, t, r, q, _ = ARGS
    lhs = bs.call_price(*ARGS) - bs.put_price(*ARGS)
    assert lhs == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t))


def test_price_dispatch_and_bad_option_type():
    assert bs.price("call", *ARGS) == bs.call_price(*ARGS)
    assert bs.price("put", *ARGS) == bs.put_price(*ARGS)
    with pytest.raises(ValueError):
        bs.price("straddle", *ARGS)


def test_deltas_differ_by_the_dividend_discount():
    t, q = ARGS[2], ARGS[4]
    gap = bs.delta("call", *ARGS) - bs.delta("put", *ARGS)
    assert gap == pytest.approx(math.exp(-q * t))


def test_greeks_match_finite_differences_of_the_price():
    s, k, t, r, q, sigma = ARGS
    h = 1e-3
    up, down = bs.call_price(s + h, k, t, r, q, sigma), bs.call_price(s - h, k, t, r, q, sigma)
    mid = bs.call_price(*ARGS)
    assert bs.delta("call", *ARGS) == pytest.approx((up - down) / (2 * h), rel=1e-6)
    assert bs.gamma(*ARGS) == pytest.approx((up - 2 * mid + down) / h**2, rel=1e-4)
    v_up, v_down = bs.call_price(s, k, t, r, q, sigma + h), bs.call_price(s, k, t, r, q, sigma - h)
    assert bs.vega(*ARGS) == pytest.approx((v_up - v_down) / (2 * h), rel=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_is_minus_the_derivative_with_respect_to_expiry(option_type):
    s, k, t, r, q, sigma = ARGS
    h = 1e-5
    longer = bs.price(option_type, s, k, t + h, r, q, sigma)
    shorter = bs.price(option_type, s, k, t - h, r, q, sigma)
    assert bs.theta(option_type, *ARGS) == pytest.approx(-(longer - shorter) / (2 * h), rel=1e-5)


@pytest.mark.parametrize(
    "bad", [(0, 105, 1, 0.05, 0.02, 0.25), (100, -1, 1, 0.05, 0.02, 0.25),
            (100, 105, 0, 0.05, 0.02, 0.25), (100, 105, 1, 0.05, 0.02, 0.0)]
)
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        bs.call_price(*bad)


def test_one_fixing_geometric_asian_is_the_vanilla_option():
    s, k, t, r, q, sigma = ARGS
    for kind in ("call", "put"):
        assert bs.geometric_asian_price(kind, s, k, t, r, q, sigma, 1) == pytest.approx(
            bs.price(kind, *ARGS), rel=1e-12
        )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_geometric_asian_matches_simulation(option_type):
    s, k, t, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
    n = 12
    paths = simulate_gbm_paths(s, r - q, sigma, t, n, 400_000, rng=5)
    geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    payoff = np.maximum(geo - k, 0) if option_type == "call" else np.maximum(k - geo, 0)
    mc = math.exp(-r * t) * payoff.mean()
    error = math.exp(-r * t) * payoff.std() / math.sqrt(len(payoff))
    exact = bs.geometric_asian_price(option_type, s, k, t, r, q, sigma, n)
    assert abs(mc - exact) < 4 * error
