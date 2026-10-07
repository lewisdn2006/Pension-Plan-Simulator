import math

import pandas as pd
import pytest

from mcengine import black_scholes as bs
from mcengine.implied_vol import build_vol_surface, solve_iv


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("strike", [70.0, 100.0, 140.0])
def test_round_trip_recovers_the_volatility(option_type, sigma, strike):
    price = bs.price(option_type, 100, strike, 0.75, 0.03, 0.01, sigma)
    if bs.vega(100, strike, 0.75, 0.03, 0.01, sigma) < 1e-3:
        pytest.skip("price barely changes with volatility here, so volatility cannot be recovered")
    assert solve_iv(price, 100, strike, 0.75, 0.03, 0.01, option_type) == pytest.approx(sigma, rel=1e-6)


def test_price_below_intrinsic_value_is_rejected():
    # Deep in the money call: it cannot be worth less than S * exp(-q T) - K * exp(-r T).
    with pytest.raises(ValueError):
        solve_iv(1.0, 150, 100, 1.0, 0.05, 0.0, "call")


def test_price_above_the_highest_possible_is_rejected():
    with pytest.raises(ValueError):
        solve_iv(1000.0, 100, 100, 1.0, 0.05, 0.0, "call")


def test_vol_surface_adds_a_column_without_changing_the_input():
    quotes = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "expiry": [0.5, 0.5, 1.0],
            "type": ["call", "put", "call"],
            "price": [bs.call_price(100, 90, 0.5, 0.05, 0.0, 0.3),
                      bs.put_price(100, 100, 0.5, 0.05, 0.0, 0.25),
                      bs.call_price(100, 110, 1.0, 0.05, 0.0, 0.22)],
        }
    )
    surface = build_vol_surface(quotes, 100.0, 0.05, 0.0)
    assert list(surface["implied_vol"].round(6)) == [0.3, 0.25, 0.22]
    assert "implied_vol" not in quotes.columns


def test_vol_surface_marks_an_impossible_quote_as_nan():
    quotes = pd.DataFrame({"strike": [100.0], "expiry": [1.0], "type": ["call"], "price": [-1.0]})
    assert math.isnan(build_vol_surface(quotes, 100.0, 0.05, 0.0).loc[0, "implied_vol"])
