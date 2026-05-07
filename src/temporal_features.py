"""Fractional Occupancy and Transition Probability — Jun 2022's two temporal phenotypes."""
from __future__ import annotations

import numpy as np


def fractional_occupancy(states: np.ndarray, K: int) -> np.ndarray:
    """Length-K vector: fraction of timepoints assigned to each state."""
    counts = np.bincount(states, minlength=K).astype(float)
    return counts / counts.sum()


def transition_probability(states: np.ndarray, K: int) -> np.ndarray:
    """K x K row-stochastic transition matrix from a 1D state sequence.

    TP[i, j] = P(state_{t+1} = j | state_t = i). Row sums to 1; states never
    visited get a uniform row to avoid NaN propagation downstream.
    """
    counts = np.zeros((K, K), dtype=float)
    if states.size < 2:
        return np.full((K, K), 1.0 / K)
    np.add.at(counts, (states[:-1], states[1:]), 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    out = np.where(row_sums > 0, counts / np.maximum(row_sums, 1), 1.0 / K)
    return out


def state_connectivity(ts: np.ndarray, states: np.ndarray, K: int) -> np.ndarray:
    """For each state, mean Pearson FC matrix across timepoints assigned to that state.

    Returns array of shape (K, n_components, n_components). States with <2
    occupied timepoints get a zero matrix (with 1s on the diagonal) to keep
    downstream Hungarian matching well-defined.
    """
    n_components = ts.shape[1]
    out = np.zeros((K, n_components, n_components), dtype=float)
    for k in range(K):
        mask = states == k
        if mask.sum() < 2:
            out[k] = np.eye(n_components)
            continue
        out[k] = np.corrcoef(ts[mask], rowvar=False)
    return out


def upper_tri_flatten(fc: np.ndarray) -> np.ndarray:
    """Flatten upper-triangular (off-diagonal) of an FC matrix to a 1D vector."""
    n = fc.shape[-1]
    iu, ju = np.triu_indices(n, k=1)
    return fc[..., iu, ju]
