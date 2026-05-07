"""Extension β — task vs rest state repertoire.

Two analyses:
  (a) Decode task timeseries through the rest-fit HMM (assumes a shared
      state repertoire across cognitive contexts; tests how rest states
      are recruited under task).
  (b) Fit an independent HMM on task data and Hungarian-match its states
      to the rest-fit states (tests whether the rest repertoire actually
      spans the task repertoire, or task elicits distinct states).

Hungarian matching is delegated to parcellation_robustness.hungarian_match_same_dim
since both sides share component count (we use the same d=300 / 139-component
parcellation for rest and task).
"""
from __future__ import annotations

import numpy as np

from .hmm_fit import FittedHMM, decode_subject, fit_group_hmm
from .parcellation_robustness import hungarian_match_same_dim, reorder_states


def decode_task_with_rest_hmm(rest_model: FittedHMM, task_ts: np.ndarray) -> np.ndarray:
    """Decode task timeseries through the rest-fit HMM. Returns (T,) state indices."""
    return decode_subject(rest_model, task_ts)


def fit_task_hmm(task_ts_concat: np.ndarray, lengths: list[int] | None, K: int,
                 seed: int = 42, n_iter: int = 100) -> FittedHMM:
    """Fit an independent K-state HMM on concatenated task timeseries."""
    return fit_group_hmm(task_ts_concat, lengths, K, seed=seed, n_iter=n_iter)


def compare_rest_vs_task_states(rest_state_fcs: np.ndarray, task_state_fcs: np.ndarray
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian-match task states to rest states. Returns (perm, similarity_matrix).

    similarity_matrix is the *unpermuted* K x K spatial-correlation matrix —
    rows = rest states, cols = task states (Figure 8 input). Permuting task
    states by perm gives matched ordering.
    """
    from .temporal_features import upper_tri_flatten
    K = rest_state_fcs.shape[0]
    flat_r = upper_tri_flatten(rest_state_fcs)
    flat_t = upper_tri_flatten(task_state_fcs)
    sim = np.corrcoef(np.vstack([flat_r, flat_t]))[:K, K:]
    perm, _ = hungarian_match_same_dim(rest_state_fcs, task_state_fcs)
    return perm, sim
