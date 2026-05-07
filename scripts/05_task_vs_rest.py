"""Extension β — task vs rest state repertoire (intended real-data entry point).

Stub. Real-data execution gated on PTN access AND on Lucas confirming at
Checkpoint 3 whether HCP task fMRI is available in d=300 ICA parcellation
or via projection through the group-ICA spatial maps from CIFTI/volumetric.

Intended behavior:
    1. Load WM and LANGUAGE task timeseries (LR + RL phase encoding each)
       in the same d=300 ICA / 139-component parcellation as rest. If d=300
       task timeseries are not available, project task CIFTI/volumetric
       data through the group-ICA spatial maps.
    2. (a) Decode task timeseries through the rest-fit HMM from Layer 1 via
       src.task_rest_comparison.decode_task_with_rest_hmm; compute per-subject
       FO during WM and LANGUAGE.
    3. Compare to rest FO with paired Wilcoxon signed-rank tests, no multiple
       comparison correction (flag as exploratory). Render Figure 7.
    4. (b) Concatenate WM + LANGUAGE timeseries and fit an independent K=4
       task HMM via src.task_rest_comparison.fit_task_hmm. Compute per-state
       mean Pearson FC.
    5. Hungarian-match the task-HMM states to rest-HMM states; render the
       full K×K spatial-correlation similarity matrix as Figure 8 (rows =
       rest states, cols = task states).

Subject count for β can drop to 5–10 if task data bandwidth is tight (decided
at Checkpoint 3); cfg.n_subjects_task controls this.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Real-data execution pending HCP S1200 PTN access + Checkpoint 3 "
        "decision on task d=300 availability and subject count. "
        "Pipeline is validated on synthetic data via scripts/synthetic_smoke.py."
    )
