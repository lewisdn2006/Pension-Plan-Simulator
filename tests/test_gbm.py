import numpy as np
import pytest

from mcengine.gbm import simulate_gbm_paths


def test_shape_and_starting_column():
    paths = simulate_gbm_paths(50.0, 0.05, 0.2, 2.0, 24, 100, rng=0)
    assert paths.shape == (100, 25)
    assert np.all(paths[:, 0] == 50.0)


def test_same_seed_same_paths_and_different_seed_different_paths():
    a = simulate_gbm_paths(1.0, 0.05, 0.2, 1.0, 12, 50, rng=3)
    b = simulate_gbm_paths(1.0, 0.05, 0.2, 1.0, 12, 50, rng=3)
    c = simulate_gbm_paths(1.0, 0.05, 0.2, 1.0, 12, 50, rng=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generator_is_used_not_copied():
    rng = np.random.default_rng(1)
    first = simulate_gbm_paths(1.0, 0.0, 0.2, 1.0, 5, 10, rng=rng)
    second = simulate_gbm_paths(1.0, 0.0, 0.2, 1.0, 5, 10, rng=rng)
    assert not np.array_equal(first, second)


def test_prices_stay_positive_even_with_huge_volatility():
    paths = simulate_gbm_paths(1.0, 0.0, 3.0, 5.0, 60, 2000, rng=0)
    assert np.all(paths > 0)


def test_zero_volatility_follows_the_drift_exactly():
    paths = simulate_gbm_paths(10.0, 0.04, 1e-12, 3.0, 36, 4, rng=0)
    assert np.allclose(paths[:, -1], 10.0 * np.exp(0.04 * 3.0))


def test_final_price_has_the_right_mean_and_log_variance():
    s0, drift, sigma, horizon = 100.0, 0.03, 0.25, 2.0
    paths = simulate_gbm_paths(s0, drift, sigma, horizon, 10, 200_000, rng=11)
    final = paths[:, -1]
    standard_error = final.std() / np.sqrt(len(final))
    assert abs(final.mean() - s0 * np.exp(drift * horizon)) < 4 * standard_error
    log_returns = np.log(final / s0)
    assert log_returns.var() == pytest.approx(sigma**2 * horizon, rel=0.02)
    assert log_returns.mean() == pytest.approx((drift - sigma**2 / 2) * horizon, abs=0.005)


def test_number_of_steps_does_not_change_the_final_distribution():
    kwargs = dict(s0=1.0, drift=0.05, sigma=0.3, horizon=1.0, n_paths=100_000)
    one = np.log(simulate_gbm_paths(steps=1, rng=1, **kwargs)[:, -1])
    many = np.log(simulate_gbm_paths(steps=50, rng=2, **kwargs)[:, -1])
    assert one.mean() == pytest.approx(many.mean(), abs=0.01)
    assert one.std() == pytest.approx(many.std(), rel=0.02)


def test_antithetic_halves_mirror_each_other_in_log_space():
    s0, drift, sigma, horizon, steps = 5.0, 0.02, 0.3, 1.0, 8
    paths = simulate_gbm_paths(s0, drift, sigma, horizon, steps, 10, antithetic=True, rng=0)
    log_ratio = np.log(paths / s0)
    centre = (drift - sigma**2 / 2) * horizon * np.arange(steps + 1) / steps
    assert np.allclose(log_ratio[:5] + log_ratio[5:], 2 * centre)


def test_antithetic_needs_an_even_number_of_paths():
    with pytest.raises(ValueError):
        simulate_gbm_paths(1.0, 0.0, 0.2, 1.0, 4, 5, antithetic=True)
