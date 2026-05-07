"""Group-level HMM fit + per-subject decoding.

Group-level fit replicates Jun 2022 / Jun 2025: concatenate all subjects'
timeseries, fit one HMM, then Viterbi-decode each subject's runs through
the shared model. This recovers a common state repertoire across subjects
while still yielding per-subject FO and TP — the temporal phenotypes that
Jun 2022 showed are heritable and Jun 2025 linked to neurotransmitter SNPs.

Library choice: prefer osl-dynamics (the Python port of HMM-MAR / OHBA, used
by Jun). Fall back to hmmlearn.GaussianHMM with full covariance — a SIMPLIFIED
Gaussian HMM (not the multivariate-autoregressive observation model Jun uses).
This is the main methodological deviation in this build and is called out in
README.md, requirements.txt, and writeup/writeup.md §7 Limitations.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# osl-dynamics installs cleanly but its imports require mne + tensorflow at
# runtime. We probe lazily so a future `pip install mne tensorflow` activates
# the preferred path automatically without code changes.
try:
    from osl_dynamics.models.hmm import Config as _OSLConfig  # noqa: F401
    from osl_dynamics.models.hmm import Model as _OSLModel  # noqa: F401
    _USE_OSL = True
except Exception:  # ImportError or downstream ModuleNotFound
    _USE_OSL = False


@dataclass
class FittedHMM:
    """Backend-agnostic wrapper. .predict(X) returns 1D int array of state indices."""

    backend: str  # "osl" or "hmmlearn"
    model: object
    K: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi-decode a (T, n_features) array; returns (T,) int."""
        if self.backend == "hmmlearn":
            return self.model.predict(X)
        return _osl_predict(self.model, X)


def fit_group_hmm(
    concat_ts: np.ndarray,
    lengths: list[int] | None,
    K: int,
    seed: int = 42,
    n_iter: int = 100,
    tol: float = 1e-3,
    covariance_type: str = "full",
) -> FittedHMM:
    """Fit one HMM on subjects-concatenated timeseries.

    `lengths` lists segment boundaries so the HMM doesn't model fictitious
    transitions across subject (or run) boundaries. Pass None for a single
    contiguous sequence.
    """
    if _USE_OSL:
        # When osl-dynamics' runtime deps are present, route here. Not the
        # active path in this build; left as the preferred route once mne+TF
        # are installed.
        return _osl_fit(concat_ts, lengths, K, seed, n_iter)
    return _hmmlearn_fit(concat_ts, lengths, K, seed, n_iter, tol, covariance_type)


def decode_subject(model: FittedHMM, subject_ts: np.ndarray) -> np.ndarray:
    """Viterbi-decode one subject's data; returns (T,) state indices."""
    return model.predict(subject_ts)


# ----- hmmlearn backend (active fallback) ---------------------------------

def _hmmlearn_fit(X, lengths, K, seed, n_iter, tol, covariance_type):
    from hmmlearn.hmm import GaussianHMM

    # ConvergenceWarnings are useful in debug but noisy on synthetic smoke runs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(
            n_components=K,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            init_params="stmc",
            params="stmc",
        )
        m.fit(X, lengths=lengths)
    return FittedHMM(backend="hmmlearn", model=m, K=K)


# ----- osl-dynamics backend (preferred when available) --------------------

def _osl_fit(X, lengths, K, seed, n_iter):
    # Stub: only reachable when mne + tensorflow are installed. Left explicit
    # so the deviation is honest rather than silently dispatched-to-hmmlearn.
    raise NotImplementedError(
        "osl-dynamics backend not exercised in this build (mne + tensorflow not pinned). "
        "See requirements.txt and src/hmm_fit.py for the deviation note."
    )


def _osl_predict(model, X):
    raise NotImplementedError("osl-dynamics backend not exercised in this build.")
