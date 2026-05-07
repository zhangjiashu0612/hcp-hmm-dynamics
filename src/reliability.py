"""Extension γ — test-retest reliability of FO and TP.

Why this matters: Jun 2022 framed FO and TP as candidate endophenotypes of
dynamic FC. Endophenotype validity requires demonstrable within-subject
stability (test-retest reliability) before heritability or genotype-effect
claims (Jun 2025) are interpretable. This module provides that check on
the same 30-subject set, splitting HCP's 4 runs into Day 1 (REST1_*) and
Day 2 (REST2_*).

Family structure note: HCP has many twins. Within-subject Day 1 vs Day 2
correlations don't depend on family structure, so it is safely ignored at
this scope. Jun's heritability analyses use a kinship matrix in mixed-
effects models — acknowledged in the write-up but not implemented here.

ICC choice: ICC(3,1) — two-way mixed, single rater, consistency — per
Shrout & Fleiss 1979. Each run is treated as an "occasion." Cicchetti 1994
thresholds: > 0.75 excellent, > 0.60 good, > 0.40 fair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .temporal_features import fractional_occupancy, transition_probability


# Mapping of run name → which day it belongs to. HCP collected REST1 on day 1
# and REST2 on day 2.
_DAY1_RUNS = ("REST1_LR", "REST1_RL")
_DAY2_RUNS = ("REST2_LR", "REST2_RL")


def features_per_day(per_run_states: dict[str, np.ndarray], K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (FO_day1, FO_day2, TP_day1_flat, TP_day2_flat) for one subject.

    `per_run_states` maps each run name to the decoded 1D state sequence for
    that run. TP is returned flattened (length K*K) to ease downstream stacking.
    """
    d1_states = np.concatenate([per_run_states[r] for r in _DAY1_RUNS])
    d2_states = np.concatenate([per_run_states[r] for r in _DAY2_RUNS])
    fo1 = fractional_occupancy(d1_states, K)
    fo2 = fractional_occupancy(d2_states, K)
    tp1 = transition_probability(d1_states, K).reshape(-1)
    tp2 = transition_probability(d2_states, K).reshape(-1)
    return fo1, fo2, tp1, tp2


def within_subject_corr(d1: np.ndarray, d2: np.ndarray) -> float:
    """Pearson correlation between two feature vectors. Returns 0 on degenerate input."""
    if d1.std() == 0 or d2.std() == 0:
        return 0.0
    return float(np.corrcoef(d1, d2)[0, 1])


def all_within_subject(d1_per_subject: np.ndarray, d2_per_subject: np.ndarray) -> np.ndarray:
    """One Pearson r per subject between Day 1 and Day 2 feature vectors."""
    return np.array([within_subject_corr(d1_per_subject[i], d2_per_subject[i])
                     for i in range(len(d1_per_subject))])


def all_between_subject(d1_per_subject: np.ndarray, d2_per_subject: np.ndarray) -> np.ndarray:
    """All cross-subject A-day1 vs B-day2 correlations (off-diagonal of subj × subj)."""
    n = len(d1_per_subject)
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            out.append(within_subject_corr(d1_per_subject[i], d2_per_subject[j]))
    return np.array(out)


def icc_3_1_per_feature(features_subj_by_run: np.ndarray) -> np.ndarray:
    """ICC(3,1) per feature column.

    Input shape: (n_subjects, n_runs, n_features). Returns length-n_features array.
    Uses pingouin.intraclass_corr; selects Type == 'ICC3' (single rater, two-way
    mixed, consistency) per Shrout & Fleiss 1979.
    """
    import pingouin as pg

    n_subj, n_runs, n_feat = features_subj_by_run.shape
    # pingouin labels Shrout & Fleiss ICC(3,1) as "ICC(C,1)" (consistency,
    # two-way mixed, single rater). See pingouin.intraclass_corr docs.
    target_type = "ICC(C,1)"
    out = np.zeros(n_feat)
    for f in range(n_feat):
        rows = []
        for s in range(n_subj):
            for r in range(n_runs):
                rows.append({"subject": s, "run": r, "value": features_subj_by_run[s, r, f]})
        df = pd.DataFrame(rows)
        if df["value"].std() == 0:
            out[f] = 0.0
            continue
        res = pg.intraclass_corr(data=df, targets="subject", raters="run",
                                 ratings="value", nan_policy="omit")
        row = res[res["Type"] == target_type]
        out[f] = float(row["ICC"].iloc[0]) if len(row) else 0.0
    return out
