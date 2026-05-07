"""All figure functions. Each figure is saved as both .png (for README) and .pdf (vector)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_dual(fig, out_path: Path) -> None:
    """Save fig as both .png and .pdf (out_path passed without extension)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _sym_vmax(arr: np.ndarray) -> float:
    """Max absolute value across an array, clipped at 1 for correlation-bounded data."""
    v = float(np.nanmax(np.abs(arr)))
    return min(max(v, 0.05), 1.0)


# ----- Figure 1: K state connectivity matrices ----------------------------

def figure1_state_matrices(state_fcs: np.ndarray, out_path: Path, title: str = "") -> None:
    """Grid of K state connectivity matrices (RdBu_r, symmetric range)."""
    K = state_fcs.shape[0]
    nrows = int(math.ceil(K / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(8, 4 * nrows), squeeze=False)
    vmax = _sym_vmax(state_fcs)
    for k, ax in enumerate(axes.flat):
        if k >= K:
            ax.axis("off")
            continue
        im = ax.imshow(state_fcs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"State {k + 1}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, label="Pearson r")
    if title:
        fig.suptitle(title)
    save_dual(fig, out_path)


# ----- Figure 2: state activation ribbon for one example subject ----------

def figure2_state_ribbon(states_one_subject: np.ndarray, K: int, out_path: Path,
                         title: str = "Example subject — state time course") -> None:
    """Stacked colored ribbon plot of per-timepoint state assignments (Jun 2025 Fig 1B style)."""
    fig, ax = plt.subplots(figsize=(12, 2.2))
    palette = sns.color_palette("tab10", n_colors=K)
    T = len(states_one_subject)
    # Encode each timepoint as a thin vertical strip colored by state.
    for k in range(K):
        mask = states_one_subject == k
        ax.fill_between(np.arange(T), 0, 1, where=mask, color=palette[k], step="post",
                        label=f"State {k + 1}")
    ax.set_xlim(0, T); ax.set_ylim(0, 1)
    ax.set_xlabel("Timepoint"); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=K, fontsize=8, frameon=False, bbox_to_anchor=(1, 1.4))
    save_dual(fig, out_path)


# ----- Figure 3: FO violin/strip across subjects --------------------------

def figure3_fo_violin(fo_per_subject: np.ndarray, out_path: Path,
                      title: str = "Fractional occupancy across subjects") -> None:
    """fo_per_subject: shape (n_subjects, K). Violin + overlaid strip."""
    n_subjects, K = fo_per_subject.shape
    long = []
    for k in range(K):
        for v in fo_per_subject[:, k]:
            long.append({"State": f"State {k + 1}", "FO": v})
    import pandas as pd
    df = pd.DataFrame(long)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x="State", y="FO", hue="State", inner=None, ax=ax,
                   palette=sns.color_palette("tab10", n_colors=K), cut=0, legend=False)
    sns.stripplot(data=df, x="State", y="FO", color="black", size=2.5, alpha=0.5, ax=ax)
    ax.set_ylim(0, max(0.6, df["FO"].max() * 1.1))
    ax.set_title(title)
    save_dual(fig, out_path)


# ----- Figure 4: mean TP heatmap ------------------------------------------

def figure4_tp_heatmap(mean_tp: np.ndarray, out_path: Path,
                       title: str = "Mean transition probability") -> None:
    """K x K heatmap of mean transition matrix across subjects."""
    K = mean_tp.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_tp, cmap="viridis", vmin=0, vmax=mean_tp.max())
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{mean_tp[i, j]:.2f}", ha="center", va="center",
                    color="white" if mean_tp[i, j] < mean_tp.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_xlabel("To"); ax.set_ylabel("From")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.045, label="P")
    save_dual(fig, out_path)


# ----- Figure 5: test-retest reliability (γ) ------------------------------

def figure5_reliability(within_corrs_fo: np.ndarray, between_corrs_fo: np.ndarray,
                        within_corrs_tp: np.ndarray, between_corrs_tp: np.ndarray,
                        icc_fo: np.ndarray, icc_tp: np.ndarray, out_path: Path) -> None:
    """Two-panel: (a) within vs between distributions for FO and TP; (b) ICC bars + heatmap."""
    K = len(icc_fo)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # (a, top-left) within vs between for FO
    ax = axes[0, 0]
    ax.hist(between_corrs_fo, bins=30, alpha=0.5, label="Between-subject", color="grey")
    ax.hist(within_corrs_fo, bins=15, alpha=0.7, label="Within-subject (Day1↔Day2)", color="C0")
    ax.set_title("FO: within vs between"); ax.set_xlabel("Pearson r"); ax.legend()
    # (a, top-right) within vs between for TP
    ax = axes[0, 1]
    ax.hist(between_corrs_tp, bins=30, alpha=0.5, label="Between-subject", color="grey")
    ax.hist(within_corrs_tp, bins=15, alpha=0.7, label="Within-subject (Day1↔Day2)", color="C1")
    ax.set_title("TP: within vs between"); ax.set_xlabel("Pearson r"); ax.legend()
    # (b, bottom-left) ICC bar plot for FO
    ax = axes[1, 0]
    ax.bar(np.arange(K), icc_fo, color=sns.color_palette("tab10", n_colors=K))
    for thr, label in [(0.4, "fair"), (0.6, "good"), (0.75, "excellent")]:
        ax.axhline(thr, ls="--", lw=0.7, color="grey")
        ax.text(K - 0.2, thr + 0.01, label, fontsize=7, color="grey", ha="right")
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_ylim(0, 1); ax.set_ylabel("ICC(3,1)"); ax.set_title("FO ICC per state")
    # (b, bottom-right) ICC heatmap for TP elements
    ax = axes[1, 1]
    im = ax.imshow(icc_tp.reshape(K, K), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_yticks(range(K)); ax.set_yticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_title("TP element ICC(3,1)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    save_dual(fig, out_path)


# ----- Figure 6: parcellation robustness (α) ------------------------------

def figure6_parcellation(ica_state_fcs: np.ndarray, schaefer_state_fcs_matched: np.ndarray,
                         matched_corrs: np.ndarray, out_path: Path) -> None:
    """2 x K side-by-side: ICA-300 (top) vs Schaefer-200 (bottom, Hungarian-reordered).

    aspect='auto' renders each FC matrix into its full subplot box regardless
    of region count (ICA ≠ Schaefer dim), keeping all 2K panels uniformly sized.
    constrained_layout handles the two-line bottom titles without overlap.
    """
    K = ica_state_fcs.shape[0]
    fig, axes = plt.subplots(2, K, figsize=(3 * K, 7.2), constrained_layout=True)
    vmax = _sym_vmax(np.concatenate([ica_state_fcs.ravel(), schaefer_state_fcs_matched.ravel()]))
    for k in range(K):
        axes[0, k].imshow(ica_state_fcs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        axes[0, k].set_title(f"ICA-300 · S{k + 1}", fontsize=10)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].imshow(schaefer_state_fcs_matched[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                          aspect="auto")
        axes[1, k].set_title(f"Schaefer-200 · S{k + 1}\n(r={matched_corrs[k]:.2f})", fontsize=10)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    fig.suptitle(f"Parcellation robustness — mean matched-pair r = {matched_corrs.mean():.2f}",
                 fontsize=12)
    save_dual(fig, out_path)


# ----- Figure 7: rest vs task FO bars -------------------------------------

def figure7_task_rest_fo(fo_rest: np.ndarray, fo_wm: np.ndarray, fo_lang: np.ndarray,
                         out_path: Path) -> None:
    """Per-state grouped bar plot: rest vs WM vs LANGUAGE, mean ± SE across subjects."""
    K = fo_rest.shape[1]
    means = np.stack([fo_rest.mean(0), fo_wm.mean(0), fo_lang.mean(0)])
    sems = np.stack([fo_rest.std(0) / np.sqrt(fo_rest.shape[0]),
                     fo_wm.std(0) / np.sqrt(fo_wm.shape[0]),
                     fo_lang.std(0) / np.sqrt(fo_lang.shape[0])])
    width = 0.27
    x = np.arange(K)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Rest", "WM", "LANGUAGE"]
    colors = ["#377eb8", "#e41a1c", "#4daf4a"]
    for i, (m, s, lbl, c) in enumerate(zip(means, sems, labels, colors)):
        ax.bar(x + (i - 1) * width, m, width, yerr=s, label=lbl, color=c, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels([f"S{k + 1}" for k in range(K)])
    ax.set_ylabel("Fractional occupancy"); ax.set_title("Rest vs task FO (rest-fit HMM decoded)")
    ax.legend()
    save_dual(fig, out_path)


# ----- Figure 8: rest vs task state similarity heatmap --------------------

def figure8_task_rest_similarity(similarity_matrix: np.ndarray, out_path: Path,
                                  title: str = "Rest-HMM vs task-HMM state similarity") -> None:
    """Rows = rest states, cols = task-HMM states. Cell = spatial r of state FCs."""
    K_r, K_t = similarity_matrix.shape
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(similarity_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(K_r):
        for j in range(K_t):
            ax.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_xticks(range(K_t)); ax.set_xticklabels([f"T{j + 1}" for j in range(K_t)])
    ax.set_yticks(range(K_r)); ax.set_yticklabels([f"R{i + 1}" for i in range(K_r)])
    ax.set_xlabel("Task-HMM state"); ax.set_ylabel("Rest-HMM state")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.045, label="Spatial r")
    save_dual(fig, out_path)
