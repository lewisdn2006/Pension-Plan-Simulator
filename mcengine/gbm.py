"""One geometric Brownian motion (GBM) path simulator, shared by everything in this package.

The option pricers call it with the risk-neutral drift r - q and the pension model calls it with
a real-world growth rate mu. The simulator does not know which it has been given: the caller
chooses the drift.
"""

from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    s0: float,
    drift: float,
    sigma: float,
    horizon: float,
    steps: int,
    n_paths: int,
    antithetic: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate paths of geometric Brownian motion, dS = drift * S dt + sigma * S dW.

    Each step uses the exact solution of the equation,

        S(t + dt) = S(t) * exp((drift - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),   Z ~ N(0, 1),

    and not the simpler Euler step S(t) + drift * S(t) * dt + sigma * S(t) * sqrt(dt) * Z.
    The exact step has no discretisation error, so the distribution of S(horizon) is the same
    for any number of steps. It also keeps the price positive on every path, because a positive
    price is multiplied by an exponential. The Euler step can go negative when Z is very low.

    The caller chooses the drift. For pricing an option use the risk-neutral drift r - q; for
    forecasting real outcomes (the pension model) use the expected growth rate mu.

    Args:
        s0: starting price.
        drift: annual drift of the price (see above).
        sigma: annual volatility.
        horizon: length of the simulation in years.
        steps: number of equal time steps, so dt = horizon / steps.
        n_paths: number of paths. Must be even when antithetic is True.
        antithetic: if True, draw Z for the first half of the paths and use -Z for the second
            half, so row i and row i + n_paths / 2 are mirror images of each other in log space.
        rng: a numpy Generator, an integer seed, or None for a fresh unseeded generator.

    Returns:
        An array of shape (n_paths, steps + 1). Row i is one path; column j is the price at
        time j * dt, so column 0 is s0 everywhere.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError("Antithetic sampling needs an even number of paths.")

    # default_rng returns an existing Generator unchanged and turns a seed or None into a new one.
    rng = np.random.default_rng(rng)

    if antithetic:
        half = rng.standard_normal((n_paths // 2, steps))
        z = np.concatenate([half, -half], axis=0)
    else:
        z = rng.standard_normal((n_paths, steps))

    dt = horizon / steps
    sqrt_dt = np.sqrt(dt)

    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = s0

    for j in range(1, steps + 1):
        dw = sqrt_dt * z[:, j - 1]
        paths[:, j] = paths[:, j - 1] * np.exp((drift - sigma**2 / 2) * dt + sigma * dw)

    return paths
