"""Extension α — ICA-300 vs Schaefer-200 state robustness.

Hungarian assignment: HMM state indices are arbitrary across separate fits,
so we match Schaefer-derived states to ICA-derived states by maximizing the
sum of spatial-similarity scores via scipy.optimize.linear_sum_assignment on
flattened upper-triangular FC matrices. Without this, a "state 1 here vs
state 1 there" comparison would be meaningless.

The Schaefer side of the pipeline produces a different number of regions
(200) than the ICA side (~139), so similarity is computed on FC topology
(matrix-shape correlation across the upper triangle), not voxel-wise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .temporal_features import upper_tri_flatten


def schaefer200_timeseries_from_cifti(cifti_path: Path, atlas_dlabel: Path) -> np.ndarray:
    """Parcellate a CIFTI dtseries with the Schaefer-200 dlabel atlas (preferred path).

    Not exercised in the synthetic smoke test. Real data path: load CIFTI dtseries,
    average across vertices grouped by Schaefer parcel ID, return (T, 200).
    """
    raise NotImplementedError(
        "Real-data CIFTI path — implement when Lucas confirms grayordinate availability "
        "at Checkpoint 3 (see data/README.md Extension α section)."
    )


def schaefer200_timeseries_from_volumetric(nii_path: Path, n_rois: int = 200,
                                           yeo_networks: int = 7,
                                           resolution_mm: int = 2) -> np.ndarray:
    """Parcellate a volumetric MNI nii.gz with Schaefer-200 (fallback path).

    Uses nilearn.NiftiLabelsMasker. Real data path; not exercised in the
    synthetic smoke test.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.maskers import NiftiLabelsMasker

    atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks,
                                      resolution_mm=resolution_mm)
    masker = NiftiLabelsMasker(labels_img=atlas["maps"], standardize=False)
    return masker.fit_transform(str(nii_path))


def hungarian_match(state_fcs_a: np.ndarray, state_fcs_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal index permutation matching b to a, by maximizing FC-topology correlation.

    state_fcs_a, state_fcs_b: (K, n_a, n_a), (K, n_b, n_b). The two sides may
    have different region counts; we match on the *shape* of FC, not on values
    at corresponding regions.

    Returns (perm, matched_corrs):
      perm[i] = index in b matched to i in a; matched_corrs[i] = correlation.
    """
    K = state_fcs_a.shape[0]
    flat_a = upper_tri_flatten(state_fcs_a)  # (K, n_a*(n_a-1)/2)
    flat_b = upper_tri_flatten(state_fcs_b)

    # Topology correlation across two arrays of (potentially) different lengths
    # is undefined unless we summarize. We summarize each state by the histogram
    # of its FC values across a fixed set of bins, then correlate histograms.
    # This makes the two sides comparable regardless of region count.
    bins = np.linspace(-1, 1, 41)

    def _hist(flat):
        out = np.zeros((flat.shape[0], len(bins) - 1))
        for k in range(flat.shape[0]):
            h, _ = np.histogram(flat[k], bins=bins, density=True)
            out[k] = h
        return out

    ha, hb = _hist(flat_a), _hist(flat_b)
    sim = np.corrcoef(np.vstack([ha, hb]))[:K, K:]  # K x K
    # Hungarian minimizes, so cost = -similarity.
    row_ind, col_ind = linear_sum_assignment(-sim)
    perm = col_ind  # perm[row_ind[i]] is the b-state matched to a-state row_ind[i]
    matched = sim[row_ind, col_ind]
    return perm, matched


def hungarian_match_same_dim(state_fcs_a: np.ndarray, state_fcs_b: np.ndarray
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match when both sides share region count (used by Extension β).

    Direct correlation on flattened upper-triangular FC.
    """
    if state_fcs_a.shape != state_fcs_b.shape:
        raise ValueError("hungarian_match_same_dim requires identical FC shapes.")
    K = state_fcs_a.shape[0]
    flat_a = upper_tri_flatten(state_fcs_a)
    flat_b = upper_tri_flatten(state_fcs_b)
    sim = np.corrcoef(np.vstack([flat_a, flat_b]))[:K, K:]
    row_ind, col_ind = linear_sum_assignment(-sim)
    return col_ind, sim[row_ind, col_ind]


def reorder_states(state_fcs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder a (K, n, n) state-FC stack along axis 0 by integer permutation."""
    return state_fcs[perm]
