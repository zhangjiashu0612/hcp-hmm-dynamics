"""Phase C — synthetic-data smoke test for the entire pipeline (all 8 figures).

Generates fake HMM-structured timeseries and runs every module end-to-end,
producing Figures 1–8 in results/figures/synthetic/. 

Run:
    python scripts/synthetic_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make `src` importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loading import load_config, standardize_timeseries  # noqa: E402
from src.hmm_fit import fit_group_hmm  # noqa: E402
from src.parcellation_robustness import hungarian_match, reorder_states  # noqa: E402
from src.plotting import (  # noqa: E402
    figure1_state_matrices,
    figure2_state_ribbon,
    figure3_fo_violin,
    figure4_tp_heatmap,
    figure5_reliability,
    figure6_parcellation,
    figure7_task_rest_fo,
    figure8_task_rest_similarity,
)
from src.reliability import (  # noqa: E402
    all_between_subject,
    all_within_subject,
    features_per_day,
    icc_3_1_per_feature,
)
from src.task_rest_comparison import compare_rest_vs_task_states, fit_task_hmm  # noqa: E402
from src.temporal_features import (  # noqa: E402
    fractional_occupancy,
    state_connectivity,
    transition_probability,
)

# Synthetic-only sizing — kept smaller than the spec's real-data shape so the
# smoke test runs in a few minutes on a laptop. The pipeline doesn't care
# about exact dimensions; it cares that everything runs end-to-end.
SYN = dict(
    n_subjects=20,
    n_runs=4,
    run_length=600,         # vs HCP's 1200
    n_components=60,        # vs HCP's 139–300
    K=4,
    schaefer_components=100,  # mock Schaefer-200 with 100 ROIs for speed
    n_subjects_task=10,
    task_run_length=dict(WM=400, LANGUAGE=316),  # mimic HCP task TRs
    seed=42,
)


# ----- Synthetic generative model -----------------------------------------

def _make_state_means_covs(K: int, D: int, rng: np.random.Generator):
    """Distinct mean vectors and PSD covariances per state."""
    means = rng.normal(0, 1.5, size=(K, D))  # well-separated
    covs = np.zeros((K, D, D))
    for k in range(K):
        # PSD = A A^T + diag perturbation; structure varies per state to give
        # distinguishable FC patterns when correlated within a state.
        A = rng.normal(0, 0.6, size=(D, max(D // 2, 4)))
        covs[k] = A @ A.T + np.eye(D) * (0.3 + 0.1 * k)
    return means, covs


def _sample_state_sequence(T: int, tp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a length-T state sequence from a transition matrix."""
    K = tp.shape[0]
    out = np.zeros(T, dtype=int)
    out[0] = rng.integers(0, K)
    for t in range(1, T):
        out[t] = rng.choice(K, p=tp[out[t - 1]])
    return out


def _sample_observations(states: np.ndarray, means: np.ndarray, covs: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Per-timepoint Gaussian sample given hidden states."""
    T, D = len(states), means.shape[1]
    out = np.zeros((T, D))
    for k in range(means.shape[0]):
        idx = np.where(states == k)[0]
        if len(idx) == 0:
            continue
        out[idx] = rng.multivariate_normal(means[k], covs[k], size=len(idx))
    return out


def generate_synthetic_dataset(n_subjects: int, n_runs: int, run_length: int,
                                D: int, K: int, seed: int) -> dict:
    """Per-subject, per-run synthetic Gaussian-HMM data with a shared state repertoire."""
    rng = np.random.default_rng(seed)
    means, covs = _make_state_means_covs(K, D, rng)
    # Group transition matrix: persistent diagonal + small off-diagonal mass.
    base_tp = np.eye(K) * 0.85 + np.ones((K, K)) * (0.15 / K)
    base_tp = base_tp / base_tp.sum(axis=1, keepdims=True)

    subjects = {}
    for s in range(n_subjects):
        # Per-subject TP perturbation (Dirichlet around group TP) → some variation
        # in temporal phenotypes across subjects (gives ICC something to chew on).
        subj_tp = np.array([rng.dirichlet(20 * base_tp[k]) for k in range(K)])
        runs = {}
        for r_idx, run_name in enumerate(["REST1_LR", "REST1_RL", "REST2_LR", "REST2_RL"][:n_runs]):
            states = _sample_state_sequence(run_length, subj_tp, rng)
            obs = _sample_observations(states, means, covs, rng)
            runs[run_name] = {"ts": obs, "states_true": states}
        subjects[s] = {"runs": runs, "tp_true": subj_tp}
    return {"subjects": subjects, "means": means, "covs": covs, "tp": base_tp, "K": K, "D": D}


# ----- Layer 1 + γ end-to-end on the primary synthetic dataset ------------

def run_layer1_and_gamma(dataset: dict, fig_dir: Path, K: int) -> dict:
    """Fit group HMM, decode, compute FO/TP/state-FC, save Figures 1–5."""
    print("[smoke] Layer 1 + γ — assembling concatenated training set")
    subjects = dataset["subjects"]
    n_subjects = len(subjects)
    run_names = list(next(iter(subjects.values()))["runs"].keys())

    # Concatenate every (subject, run) as a separate sequence so the HMM
    # doesn't model fictitious transitions across subject/run boundaries.
    chunks, lengths, owners = [], [], []  # owners: list of (subject_idx, run_name)
    for s_idx, s in subjects.items():
        for run_name, run in s["runs"].items():
            ts = standardize_timeseries(run["ts"])
            chunks.append(ts)
            lengths.append(len(ts))
            owners.append((s_idx, run_name))
    X = np.vstack(chunks)
    print(f"[smoke] training matrix shape = {X.shape}, {len(lengths)} sequences")

    t0 = time.time()
    model = fit_group_hmm(X, lengths=lengths, K=K, seed=42, n_iter=30, covariance_type="full")
    print(f"[smoke] HMM fit: {time.time() - t0:.1f}s")

    # Decode each (subject, run) separately.
    print("[smoke] decoding per-subject, per-run")
    per_subject = {s_idx: {} for s_idx in subjects}
    offset = 0
    for length, (s_idx, run_name) in zip(lengths, owners):
        seg = X[offset:offset + length]
        per_subject[s_idx][run_name] = model.predict(seg)
        offset += length

    # Per-subject FO and TP (full 4-run concat).
    fo_per = np.zeros((n_subjects, K))
    tp_per = np.zeros((n_subjects, K, K))
    for s_idx, run_states in per_subject.items():
        all_states = np.concatenate([run_states[r] for r in run_names])
        fo_per[s_idx] = fractional_occupancy(all_states, K)
        tp_per[s_idx] = transition_probability(all_states, K)

    # Group-level state connectivity: pool all timepoints across subjects/runs.
    print("[smoke] computing state connectivity matrices")
    all_states_concat = np.concatenate([per_subject[s_idx][r] for s_idx in per_subject for r in run_names])
    state_fcs = state_connectivity(X, all_states_concat, K)

    # Figure 1, 2, 3, 4
    figure1_state_matrices(state_fcs, fig_dir / "figure1_state_matrices_synthetic",
                           title="K=4 state connectivity (synthetic)")
    example_states = np.concatenate([per_subject[0][r] for r in run_names])
    figure2_state_ribbon(example_states, K, fig_dir / "figure2_state_ribbon_synthetic",
                         title="Example subject — state time course (synthetic)")
    figure3_fo_violin(fo_per, fig_dir / "figure3_fo_violin_synthetic",
                      title="FO across subjects (synthetic)")
    figure4_tp_heatmap(tp_per.mean(axis=0), fig_dir / "figure4_tp_heatmap_synthetic",
                       title="Mean transition probability (synthetic)")
    print("[smoke] Figures 1–4 saved")

    # ----- γ -----
    print("[smoke] computing test-retest reliability (γ)")
    fo_d1, fo_d2, tp_d1, tp_d2 = [], [], [], []
    fo_per_run = np.zeros((n_subjects, len(run_names), K))
    tp_per_run_flat = np.zeros((n_subjects, len(run_names), K * K))
    for s_idx, run_states in per_subject.items():
        f1, f2, t1, t2 = features_per_day(run_states, K)
        fo_d1.append(f1); fo_d2.append(f2); tp_d1.append(t1); tp_d2.append(t2)
        for r_idx, r_name in enumerate(run_names):
            fo_per_run[s_idx, r_idx] = fractional_occupancy(run_states[r_name], K)
            tp_per_run_flat[s_idx, r_idx] = transition_probability(run_states[r_name], K).reshape(-1)
    fo_d1, fo_d2 = np.array(fo_d1), np.array(fo_d2)
    tp_d1, tp_d2 = np.array(tp_d1), np.array(tp_d2)

    within_fo = all_within_subject(fo_d1, fo_d2)
    between_fo = all_between_subject(fo_d1, fo_d2)
    within_tp = all_within_subject(tp_d1, tp_d2)
    between_tp = all_between_subject(tp_d1, tp_d2)
    print(f"[smoke] FO  within mean r = {within_fo.mean():+.3f}, between mean r = {between_fo.mean():+.3f}")
    print(f"[smoke] TP  within mean r = {within_tp.mean():+.3f}, between mean r = {between_tp.mean():+.3f}")

    icc_fo = icc_3_1_per_feature(fo_per_run)
    icc_tp = icc_3_1_per_feature(tp_per_run_flat)
    print(f"[smoke] FO ICC(3,1) per state: {np.array2string(icc_fo, precision=2)}")
    print(f"[smoke] TP ICC(3,1) summary: mean = {icc_tp.mean():.2f}")

    figure5_reliability(within_fo, between_fo, within_tp, between_tp, icc_fo, icc_tp,
                        fig_dir / "figure5_reliability_synthetic")
    print("[smoke] Figure 5 saved")

    return {"model": model, "state_fcs": state_fcs, "fo_per": fo_per,
            "tp_per": tp_per, "per_subject": per_subject, "run_names": run_names}


# ----- α: Schaefer-200 mock pipeline --------------------------------------

def run_alpha(layer1: dict, fig_dir: Path, K: int) -> None:
    """Generate a synthetic Schaefer-like dataset, fit HMM, Hungarian-match to ICA states."""
    print("[smoke] α — synthetic Schaefer-like dataset")
    schaefer_ds = generate_synthetic_dataset(
        n_subjects=SYN["n_subjects"], n_runs=SYN["n_runs"], run_length=SYN["run_length"],
        D=SYN["schaefer_components"], K=K, seed=SYN["seed"] + 1,
    )
    chunks, lengths = [], []
    for s in schaefer_ds["subjects"].values():
        for run in s["runs"].values():
            ts = standardize_timeseries(run["ts"])
            chunks.append(ts); lengths.append(len(ts))
    X_sch = np.vstack(chunks)

    t0 = time.time()
    sch_model = fit_group_hmm(X_sch, lengths=lengths, K=K, seed=42, n_iter=30,
                              covariance_type="full")
    print(f"[smoke] α HMM fit: {time.time() - t0:.1f}s")
    sch_states = sch_model.predict(X_sch)
    sch_state_fcs = state_connectivity(X_sch, sch_states, K)

    perm, matched_corrs = hungarian_match(layer1["state_fcs"], sch_state_fcs)
    print(f"[smoke] α matched-pair correlations: {np.array2string(matched_corrs, precision=2)}")
    figure6_parcellation(layer1["state_fcs"], reorder_states(sch_state_fcs, perm),
                         matched_corrs, fig_dir / "figure6_parcellation_synthetic")
    print("[smoke] Figure 6 saved")


# ----- β: rest-fit decode on task + independent task HMM ------------------

def run_beta(layer1: dict, fig_dir: Path, K: int) -> None:
    """Synthetic 'task' data; rest model decoded on task; independent task HMM matched."""
    print("[smoke] β — synthetic task data")
    rest_model = layer1["model"]
    fo_rest = layer1["fo_per"][: SYN["n_subjects_task"]]

    rng_task = np.random.default_rng(SYN["seed"] + 99)
    means, covs = _make_state_means_covs(K, SYN["n_components"], rng_task)
    # Build a *task* TP biased toward different states than rest, so β has a
    # signal to detect (rest-state recruitment shifts under task).
    bias_perm = rng_task.permutation(K)
    biased_tp = np.eye(K) * 0.7 + np.eye(K)[:, bias_perm] * 0.15 + np.ones((K, K)) * (0.15 / K)
    biased_tp /= biased_tp.sum(axis=1, keepdims=True)

    task_data = {"WM": [], "LANGUAGE": []}
    task_decoded = {"WM": [], "LANGUAGE": []}
    for s in range(SYN["n_subjects_task"]):
        for task in ("WM", "LANGUAGE"):
            T = SYN["task_run_length"][task]
            states = _sample_state_sequence(T, biased_tp, rng_task)
            ts = standardize_timeseries(_sample_observations(states, means, covs, rng_task))
            task_data[task].append(ts)
            task_decoded[task].append(rest_model.predict(ts))

    fo_wm = np.array([fractional_occupancy(d, K) for d in task_decoded["WM"]])
    fo_lang = np.array([fractional_occupancy(d, K) for d in task_decoded["LANGUAGE"]])
    print(f"[smoke] β FO mean: rest={fo_rest.mean(0).round(2)}, "
          f"WM={fo_wm.mean(0).round(2)}, LANG={fo_lang.mean(0).round(2)}")

    figure7_task_rest_fo(fo_rest, fo_wm, fo_lang, fig_dir / "figure7_task_rest_fo_synthetic")
    print("[smoke] Figure 7 saved")

    # Independent task HMM (fit on WM + LANG concatenated).
    chunks, lengths = [], []
    for task_ts_list in task_data.values():
        for ts in task_ts_list:
            chunks.append(ts); lengths.append(len(ts))
    X_task = np.vstack(chunks)
    t0 = time.time()
    task_model = fit_task_hmm(X_task, lengths=lengths, K=K, seed=42, n_iter=30)
    print(f"[smoke] β task HMM fit: {time.time() - t0:.1f}s")
    task_states_all = task_model.predict(X_task)
    task_state_fcs = state_connectivity(X_task, task_states_all, K)

    _, sim = compare_rest_vs_task_states(layer1["state_fcs"], task_state_fcs)
    figure8_task_rest_similarity(sim, fig_dir / "figure8_task_rest_similarity_synthetic")
    print(f"[smoke] β rest-vs-task similarity diag mean = {np.diag(sim).mean():+.2f}")
    print("[smoke] Figure 8 saved")


# ----- main ---------------------------------------------------------------

def main():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    fig_dir = (Path(__file__).resolve().parents[1] / cfg["output"]["figures_dir"]
               / cfg["output"]["synthetic_subdir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] writing figures to {fig_dir}")
    K = SYN["K"]

    print("[smoke] generating primary synthetic dataset")
    primary = generate_synthetic_dataset(
        n_subjects=SYN["n_subjects"], n_runs=SYN["n_runs"], run_length=SYN["run_length"],
        D=SYN["n_components"], K=K, seed=SYN["seed"],
    )

    layer1 = run_layer1_and_gamma(primary, fig_dir, K)
    run_alpha(layer1, fig_dir, K)
    run_beta(layer1, fig_dir, K)

    print("[smoke] DONE — 8 synthetic figures in", fig_dir)


if __name__ == "__main__":
    main()
