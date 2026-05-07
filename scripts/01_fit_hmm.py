"""Layer 1 — group-level HMM on HCP rs-fMRI d=300 timeseries (intended real-data entry point).

Stub. Real-data execution is gated on PTN access (see data/README.md).
The pipeline below describes the intended behavior; it is fully implemented
on synthetic data in scripts/synthetic_smoke.py and uses the same src/ modules.

Intended behavior:
    1. Load config.yaml (paths, K_primary=4, HMM_SEED=42, subject list).
    2. For each subject in subject_list.txt:
         - Load the per-subject d=300 .txt via src.data_loading.load_ica300_subject.
         - Optionally subset to 139 cortical/subcortical components.
         - Split the 4800-tp file into 4 runs of 1200 via src.data_loading.split_runs.
         - Per-component z-score via standardize_timeseries.
    3. Concatenate every (subject, run) sequence with a `lengths` vector so the
       HMM does not model fictitious cross-boundary transitions, then call
       src.hmm_fit.fit_group_hmm with K=4.
    4. Decode each run separately via FittedHMM.predict; compute per-subject FO
       and TP via src.temporal_features; write fo_per_subject.csv and
       tp_per_subject.csv to results/tables/.
    5. Compute per-state mean Pearson FC across all timepoints assigned to
       that state via src.temporal_features.state_connectivity.
    6. Render Figures 1–4 to results/figures/ via src.plotting (PNG + PDF).
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Real-data execution pending HCP S1200 PTN access. "
        "Pipeline is validated on synthetic data via scripts/synthetic_smoke.py."
    )
