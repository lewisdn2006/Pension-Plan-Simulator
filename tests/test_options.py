import math

import numpy as np
import pytest

from mcengine import black_scholes as bs
from mcengine.gbm import simulate_gbm_paths
from mcengine.options import MarketData, MonteCarloEngine, OptionContract

MARKET = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02, volatility=0.25)


def exact(option_type, strike, expiry=1.0, market=MARKET):
    return bs.price(
        option_type, market.spot_price, strike, expiry,
        market.risk_free_rate, market.dividend_yield, market.volatility,
    )


# ---- inputs -----------------------------------------------------------------------------

def test_contract_and_market_validation():
    with pytest.raises(ValueError):
        OptionContract("straddle", 100, 1.0)
    with pytest.raises(ValueError):
        OptionContract("call", -1, 1.0)
    with pytest.raises(ValueError):
        OptionContract("call", 100, 0.0)
    with pytest.raises(ValueError):
        OptionContract("call", 100, 1.0, 120.0, "sideways")
    with pytest.raises(ValueError):
        MarketData(0.0, 0.05, 0.0, 0.2)
    with pytest.raises(ValueError):
        MarketData(100.0, 0.05, 0.0, 0.0)


# ---- European ---------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
@pytest.mark.parametrize("control_variate", [False, True])
def test_european_price_is_within_four_standard_errors_of_black_scholes(
    option_type, antithetic, control_variate
):
    option = OptionContract(option_type, 105.0, 1.0)
    result = MonteCarloEngine(seed=7).price_european(
        MARKET, option, num_paths=200_000, antithetic=antithetic,
        control_variate=control_variate, compute_greeks=False,
    )
    assert abs(result.price - exact(option_type, 105.0)) < 4 * result.std_error


def test_confidence_interval_is_the_price_plus_or_minus_1_96_standard_errors():
    result = MonteCarloEngine(seed=1).price_european(
        MARKET, OptionContract("call", 100, 1.0), 50_000, compute_greeks=False
    )
    low, high = result.confidence_interval
    assert low == pytest.approx(result.price - 1.96 * result.std_error, rel=1e-3)
    assert high == pytest.approx(result.price + 1.96 * result.std_error, rel=1e-3)
    assert result.num_paths == 50_000


def test_same_seed_gives_the_same_price_and_different_seeds_do_not():
    option = OptionContract("call", 100, 1.0)
    a = MonteCarloEngine(seed=3).price_european(MARKET, option, 10_000, compute_greeks=False)
    b = MonteCarloEngine(seed=3).price_european(MARKET, option, 10_000, compute_greeks=False)
    c = MonteCarloEngine(seed=4).price_european(MARKET, option, 10_000, compute_greeks=False)
    assert a.price == b.price
    assert a.price != c.price


def test_control_variate_reduces_the_standard_error():
    option = OptionContract("call", 105.0, 1.0)
    plain = MonteCarloEngine(seed=2).price_european(
        MARKET, option, 100_000, control_variate=False, compute_greeks=False)
    controlled = MonteCarloEngine(seed=2).price_european(
        MARKET, option, 100_000, control_variate=True, compute_greeks=False)
    assert controlled.std_error < 0.8 * plain.std_error


@pytest.mark.parametrize("antithetic,control_variate", [(False, False), (True, False), (False, True), (True, True)])
def test_reported_confidence_interval_covers_the_true_price_about_95_percent_of_the_time(
    antithetic, control_variate
):
    option = OptionContract("call", 105.0, 1.0)
    truth = exact("call", 105.0)
    engine = MonteCarloEngine(seed=99)
    hits = 0
    runs = 300
    for _ in range(runs):
        r = engine.price_european(
            MARKET, option, 2_000, antithetic=antithetic,
            control_variate=control_variate, compute_greeks=False)
        hits += r.confidence_interval[0] <= truth <= r.confidence_interval[1]
    assert 0.90 <= hits / runs <= 0.99


def test_put_call_parity_holds_for_the_simulated_prices():
    call = MonteCarloEngine(seed=5).price_european(
        MARKET, OptionContract("call", 100, 1.0), 400_000, compute_greeks=False)
    put = MonteCarloEngine(seed=6).price_european(
        MARKET, OptionContract("put", 100, 1.0), 400_000, compute_greeks=False)
    parity = 100 * math.exp(-0.02) - 100 * math.exp(-0.05)
    assert abs((call.price - put.price) - parity) < 4 * math.hypot(call.std_error, put.std_error)


def test_greeks_match_black_scholes():
    option = OptionContract("call", 105.0, 1.0)
    result = MonteCarloEngine(seed=8).price_european(MARKET, option, 400_000)
    args = (100.0, 105.0, 1.0, 0.05, 0.02, 0.25)
    assert result.delta == pytest.approx(bs.delta("call", *args), abs=0.01)
    assert result.gamma == pytest.approx(bs.gamma(*args), rel=0.15)
    assert result.vega == pytest.approx(bs.vega(*args), rel=0.03)


def test_put_delta_is_negative():
    result = MonteCarloEngine(seed=8).price_european(
        MARKET, OptionContract("put", 100.0, 1.0), 200_000)
    assert -1 < result.delta < 0


def test_number_of_steps_does_not_change_the_european_price():
    option = OptionContract("call", 100.0, 1.0)
    one = MonteCarloEngine(seed=1).price_european(MARKET, option, 200_000, num_steps=1, compute_greeks=False)
    many = MonteCarloEngine(seed=2).price_european(MARKET, option, 200_000, num_steps=50, compute_greeks=False)
    assert abs(one.price - many.price) < 4 * math.hypot(one.std_error, many.std_error)


# ---- Asian ------------------------------------------------------------------------------

def test_arithmetic_asian_is_worth_at_least_the_geometric_asian_for_a_call():
    option = OptionContract("call", 100.0, 1.0)
    arithmetic = MonteCarloEngine(seed=1).price_asian(MARKET, option, 200_000, num_steps=12)
    geometric = bs.geometric_asian_price("call", 100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 12)
    assert arithmetic.price > geometric


def test_asian_control_variate_agrees_with_plain_and_is_much_tighter():
    option = OptionContract("call", 100.0, 1.0)
    plain = MonteCarloEngine(seed=3).price_asian(MARKET, option, 200_000, control_variate=False)
    controlled = MonteCarloEngine(seed=4).price_asian(MARKET, option, 200_000, control_variate=True)
    assert abs(plain.price - controlled.price) < 4 * math.hypot(plain.std_error, controlled.std_error)
    assert controlled.std_error < 0.1 * plain.std_error


def test_asian_call_is_cheaper_than_the_european_call_with_the_same_strike():
    option = OptionContract("call", 100.0, 1.0)
    asian = MonteCarloEngine(seed=1).price_asian(MARKET, option, 100_000)
    assert asian.price < exact("call", 100.0)


def test_asian_with_one_fixing_is_the_european_option():
    option = OptionContract("call", 100.0, 1.0)
    asian = MonteCarloEngine(seed=1).price_asian(MARKET, option, 200_000, num_steps=1)
    assert abs(asian.price - exact("call", 100.0)) < 4 * asian.std_error + 1e-9


# ---- barrier ----------------------------------------------------------------------------

def vanilla_on_same_paths(seed, option, n_paths, n_steps):
    """Vanilla price on exactly the paths the engine will draw for its first call with `seed`."""
    path_seed = MonteCarloEngine(seed=seed)._draw_seed()
    paths = simulate_gbm_paths(
        MARKET.spot_price, MARKET.risk_free_rate - MARKET.dividend_yield, MARKET.volatility,
        option.expiry, n_steps, n_paths, rng=path_seed)
    payoff = np.maximum(paths[:, -1] - option.strike, 0)
    return math.exp(-MARKET.risk_free_rate * option.expiry) * payoff.mean()


@pytest.mark.parametrize("kind,level", [("up", 125.0), ("down", 80.0)])
def test_knock_in_plus_knock_out_equals_the_vanilla_option(kind, level):
    out = OptionContract("call", 100.0, 1.0, level, f"{kind}-and-out")
    knock_in = OptionContract("call", 100.0, 1.0, level, f"{kind}-and-in")
    price_out = MonteCarloEngine(seed=5).price_barrier(MARKET, out, 50_000, 52).price
    price_in = MonteCarloEngine(seed=5).price_barrier(MARKET, knock_in, 50_000, 52).price
    vanilla = vanilla_on_same_paths(5, OptionContract("call", 100.0, 1.0), 50_000, 52)
    assert price_out + price_in == pytest.approx(vanilla, rel=1e-12)


def test_knock_ins_are_not_priced_as_knock_outs():
    knock_in = OptionContract("call", 100.0, 1.0, 125.0, "up-and-in")
    knock_out = OptionContract("call", 100.0, 1.0, 125.0, "up-and-out")
    engine_args = dict(num_paths=50_000, num_steps=52)
    price_in = MonteCarloEngine(seed=5).price_barrier(MARKET, knock_in, **engine_args).price
    price_out = MonteCarloEngine(seed=5).price_barrier(MARKET, knock_out, **engine_args).price
    assert abs(price_in - price_out) > 0.5


def test_knock_out_is_worth_less_than_the_vanilla_option():
    option = OptionContract("call", 100.0, 1.0, 120.0, "up-and-out")
    result = MonteCarloEngine(seed=1).price_barrier(MARKET, option, 100_000, 52)
    assert 0 < result.price < exact("call", 100.0)


def test_barrier_never_reached_leaves_the_vanilla_price():
    option = OptionContract("call", 100.0, 1.0, 1e9, "up-and-out")
    result = MonteCarloEngine(seed=5).price_barrier(MARKET, option, 20_000, 20)
    assert result.price == pytest.approx(
        vanilla_on_same_paths(5, OptionContract("call", 100.0, 1.0), 20_000, 20), rel=1e-12)


def test_barrier_just_above_the_spot_kills_almost_every_path():
    option = OptionContract("call", 100.0, 1.0, 100.001, "up-and-out")
    result = MonteCarloEngine(seed=1).price_barrier(MARKET, option, 20_000, 252)
    assert result.price < 0.01


def test_barrier_on_the_wrong_side_of_the_spot_is_rejected():
    with pytest.raises(ValueError):
        MonteCarloEngine(seed=1).price_barrier(
            MARKET, OptionContract("call", 100.0, 1.0, 90.0, "up-and-out"))
    with pytest.raises(ValueError):
        MonteCarloEngine(seed=1).price_barrier(
            MARKET, OptionContract("put", 100.0, 1.0, 110.0, "down-and-in"))


def test_barrier_option_without_a_barrier_is_rejected():
    with pytest.raises(ValueError):
        MonteCarloEngine(seed=1).price_barrier(MARKET, OptionContract("call", 100.0, 1.0))


def test_down_and_in_put_is_worth_less_than_the_vanilla_put():
    option = OptionContract("put", 100.0, 1.0, 85.0, "down-and-in")
    result = MonteCarloEngine(seed=1).price_barrier(MARKET, option, 100_000, 52)
    assert 0 < result.price < exact("put", 100.0)
