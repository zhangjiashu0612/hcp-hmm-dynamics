"""Extension γ — test-retest reliability on HCP rs-fMRI (intended real-data entry point).

Stub. Real-data execution gated on PTN access (see data/README.md).

Intended behavior:
    1. Reuse the K=4 group HMM fit from 01_fit_hmm.py (load checkpoint or refit).
    2. For each subject, decode each of the 4 runs separately, then split runs
       into Day 1 (REST1_LR + REST1_RL) and Day 2 (REST2_LR + REST2_RL) per
       src.reliability.features_per_day; compute per-day FO and per-day TP.
    3. Compute within-subject Day1↔Day2 Pearson r per subject for FO and TP.
    4. Compute the full N×(N-1) between-subject distribution (subj A Day 1 vs
       subj B Day 2) via src.reliability.all_between_subject.
    5. Compute ICC(3,1) per FO state and per TP element across the 4 individual
       runs treated as occasions, via src.reliability.icc_3_1_per_feature
       (pingouin's "ICC(C,1)" — single rater, two-way mixed, consistency, per
       Shrout & Fleiss 1979). Cicchetti 1994 thresholds annotated on the figure.
    6. Render Figure 5 + write icc_fo.csv, icc_tp.csv to results/tables/.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Real-data execution pending HCP S1200 PTN access. "
        "Pipeline is validated on synthetic data via scripts/synthetic_smoke.py."
    )
