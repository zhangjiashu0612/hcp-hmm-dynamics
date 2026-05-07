"""Extension α — Schaefer-200 vs ICA-300 robustness (intended real-data entry point).

Stub. Real-data execution gated on PTN access AND on Lucas confirming at
Checkpoint 3 whether grayordinate CIFTI or volumetric MNI rs-fMRI is available.

Intended behavior:
    1. For each Layer-1 subject, parcellate rs-fMRI to Schaefer-200 (7-network):
         - Preferred: src.parcellation_robustness.schaefer200_timeseries_from_cifti
           on grayordinate `*_Atlas_MSMAll_hp2000_clean.dtseries.nii`.
         - Fallback: src.parcellation_robustness.schaefer200_timeseries_from_volumetric
           on volumetric `*_hp2000_clean.nii.gz` via nilearn.NiftiLabelsMasker
           (introduces a grayordinate-vs-volumetric mismatch with d=300 ICA;
           must be flagged in the write-up if this path is taken).
    2. Concatenate all (subject, run) Schaefer-200 timeseries with `lengths`
       and fit an independent K=4 group HMM via src.hmm_fit.fit_group_hmm.
    3. Compute per-state mean Pearson FC (200x200) for the Schaefer side.
    4. Hungarian-match Schaefer states to the ICA-300 states from Layer 1
       via src.parcellation_robustness.hungarian_match (FC-histogram-based,
       handles different region counts cleanly).
    5. Render Figure 6 + report mean matched-pair r and FO correlation
       between parcellations across subjects.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Real-data execution pending HCP S1200 PTN access + Checkpoint 3 "
        "decision on CIFTI vs volumetric source. "
        "Pipeline is validated on synthetic data via scripts/synthetic_smoke.py."
    )
