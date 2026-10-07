import numpy as np
import pytest

from mcengine import pension


def reference_fund(path, years, monthly):
    """The fund for one path, built up one payment at a time."""
    shares = 0.0
    for month in range(12 * years + 1):
        shares += monthly / path[month]
    return shares * path[12 * years]


def test_index_paths_are_monthly_and_start_at_one():
    paths = pension.simulate_index_paths(0.05, 20, 3, rng=0)
    assert paths.shape == (20, 37)
    assert np.all(paths[:, 0] == pension.S0)


def test_fund_matches_a_payment_by_payment_calculation():
    paths = pension.simulate_index_paths(0.05, 6, 5, rng=1)
    values = pension.fund_values(paths, 5, 250.0)
    for path, value in zip(paths, values):
        assert value == pytest.approx(reference_fund(path, 5, 250.0))


def test_a_flat_index_gives_one_unit_of_fund_per_payment():
    flat = np.ones((3, 25))
    assert np.allclose(pension.fund_values(flat, 2, 100.0), 25 * 100.0)


def test_fund_is_proportional_to_the_monthly_payment():
    paths = pension.simulate_index_paths(0.05, 50, 10, rng=2)
    assert np.allclose(pension.fund_values(paths, 10, 300.0), 3 * pension.fund_values(paths, 10, 100.0))


def test_fund_does_not_depend_on_the_starting_price():
    paths = pension.simulate_index_paths(0.05, 50, 10, rng=2)
    assert np.allclose(pension.unit_fund_values(paths, 10), pension.unit_fund_values(7.5 * paths, 10))


def test_fewer_years_than_the_paths_cover_uses_the_first_part():
    paths = pension.simulate_index_paths(0.05, 10, 10, rng=2)
    expected = reference_fund(paths[0], 4, 1.0)
    assert pension.unit_fund_values(paths, 4)[0] == pytest.approx(expected)


def test_asking_for_more_years_than_the_paths_cover_is_an_error():
    paths = pension.simulate_index_paths(0.05, 10, 3, rng=2)
    with pytest.raises(ValueError):
        pension.unit_fund_values(paths, 4)


def test_negative_payment_is_rejected():
    paths = pension.simulate_index_paths(0.05, 10, 3, rng=2)
    with pytest.raises(ValueError):
        pension.fund_values(paths, 3, -1.0)


def test_outcome_probabilities_on_known_values():
    # 1 year at 10 a month: 120 paid in. Doubling is above 240.
    values = np.array([50.0, 119.0, 120.0, 130.0, 241.0, 300.0, 2_500_000.0, 10.0])
    loss, double, high = pension.outcome_probabilities(values, 1, 10.0)
    assert loss == 3 / 8      # 50, 119 and 10
    assert double == 3 / 8    # 241, 300 and 2,500,000
    assert high == 1 / 8


def test_probability_curve_matches_counting_directly():
    paths = pension.simulate_index_paths(0.05, 2000, 10, rng=3)
    unit = pension.unit_fund_values(paths, 10)
    payments = np.arange(0, 600, 20.0)
    curve = pension.probability_curve(unit, payments, 100_000.0)
    brute = [np.mean(m * unit > 100_000.0) for m in payments]
    assert np.allclose(curve, brute)


def test_probability_curve_rises_with_the_payment_and_starts_at_zero():
    paths = pension.simulate_index_paths(0.05, 2000, 10, rng=3)
    unit = pension.unit_fund_values(paths, 10)
    curve = pension.probability_curve(unit, np.arange(0, 1000, 20.0), 100_000.0)
    assert curve[0] == 0.0
    assert np.all(np.diff(curve) >= 0)
    assert curve[-1] > 0.9


def test_a_fund_exactly_on_the_target_does_not_count_as_above_it():
    unit = np.array([1.0, 2.0, 3.0])
    assert pension.probability_curve(unit, [10.0], 20.0)[0] == pytest.approx(1 / 3)  # only 3 * 10 > 20


def test_higher_growth_gives_better_odds():
    low = pension.unit_fund_values(pension.simulate_index_paths(0.03, 5000, 20, rng=4), 20)
    high = pension.unit_fund_values(pension.simulate_index_paths(0.07, 5000, 20, rng=4), 20)
    p_low = pension.probability_curve(low, [500.0], 300_000.0)[0]
    p_high = pension.probability_curve(high, [500.0], 300_000.0)[0]
    assert p_high > p_low


def test_required_contribution_is_the_first_payment_that_reaches_the_target():
    payments = [0, 20, 40, 60]
    assert pension.required_contribution(payments, [0.1, 0.5, 0.95, 0.99]) == 40.0
    assert pension.required_contribution(payments, [0.1, 0.5, 0.9, 0.94]) is None
    assert pension.required_contribution(payments, [0.1, 0.5, 0.8, 0.9], target=0.8) == 40.0
