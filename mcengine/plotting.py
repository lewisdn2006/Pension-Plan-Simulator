"""Plots for the examples. Each function saves a PNG and closes the figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_paths(paths: np.ndarray, horizon: float, path, title: str, ylabel: str, n_show: int = 60) -> Path:
    """Some sample paths with the mean and the 5th and 95th percentiles."""
    t = np.linspace(0, horizon, paths.shape[1])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(t, paths[:n_show].T, color="steelblue", alpha=0.25, linewidth=0.6)
    ax.plot(t, paths.mean(axis=0), color="crimson", linewidth=2, label="mean")
    ax.plot(t, np.percentile(paths, 95, axis=0), "g--", label="95th percentile")
    ax.plot(t, np.percentile(paths, 5, axis=0), "g--", label="5th percentile")
    ax.set(xlabel="Time (years)", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_histogram(values: np.ndarray, path, title: str, xlabel: str, bins: int = 60) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set(xlabel=xlabel, ylabel="Number of paths", title=title)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_convergence(
    path_counts: Sequence[int], prices: Sequence[float], exact: Optional[float], path
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(path_counts, prices, "o-", label="Monte Carlo price")
    if exact is not None:
        ax.axhline(exact, color="crimson", linestyle="--", label=f"Black-Scholes {exact:.4f}")
    ax.set(xscale="log", xlabel="Number of paths", ylabel="Price", title="Monte Carlo convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_probability_curves(
    payments: Sequence[float], curves: dict, target_value: float, path, title: str
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, probs in curves.items():
        ax.plot(payments, probs, label=label)
    ax.axhline(0.95, color="grey", linestyle=":")
    ax.set(
        xlabel="Monthly payment",
        ylabel=f"Chance of finishing above {target_value:,.0f}",
        title=title,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)
