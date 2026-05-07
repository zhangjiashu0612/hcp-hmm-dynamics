# hcp-hmm-dynamics

HMM-based extraction of connectome state dynamics from HCP resting-state fMRI, with parcellation, task, and reliability validations.

> **Status: Synthetic-validated method demonstration.** Pipeline is data-source-agnostic; real-data execution requires HCP S1200 PTN access (currently unavailable post-BALSA migration; see [data/README.md](data/README.md)).

This is a methods-skill demonstration following the choices in Jun et al. 2022 (*NeuroImage*) and Jun et al. 2025 (*J Neurosci*) — group-level Hidden Markov Model on minimally preprocessed HCP rs-fMRI, K=4 states, no GSR, group-ICA d=300 parcellation, per-subject Fractional Occupancy and Transition Probability. Three forward extensions: parcellation robustness (Schaefer-200), task vs rest state repertoire (WM, LANGUAGE), and test-retest reliability via ICC(3,1). Not a replication (no dbGAP genetics); not novel science.

## How to run

```bash
conda create -n hcp-hmm python=3.11 -y && conda activate hcp-hmm
pip install -r requirements.txt
# Place HCP downloads as described in data/README.md, then:
python scripts/01_fit_hmm.py            # Layer 1 — K=4 primary
python scripts/02_robustness_k6.py      # K=6 robustness check
python scripts/03_test_retest.py        # Extension γ — Day1/Day2 + ICC
python scripts/04_parcellation_robustness.py   # Extension α — ICA-300 vs Schaefer-200
python scripts/05_task_vs_rest.py       # Extension β — WM, LANGUAGE
```

All scripts read `config.yaml` for paths, K values, and `HMM_SEED=42`.

## Key findings (synthetic validation)

- 8 figures rendered end-to-end on synthetic data via `scripts/synthetic_smoke.py`.
- ICC(3,1) FO range **[0.68, 0.75, 0.84, 0.91]** (fair → excellent per Cicchetti 1994).
- ICC(3,1) TP mean **0.47** (fair).
- State persistence: TP diagonal **0.87–0.91** (by-design property of synthetic generator).
- *Caveat:* these are method-validation values on synthetic data, not empirical findings.

## Main figures

> *Figures 1–4 (Layer 1) embedded here after Phase D. Validations paragraph below links to Figures 5–8.*

- **Figure 1** — K=4 state connectivity matrices.
- **Figure 2** — example-subject state activation ribbon plot.
- **Figure 3** — FO distribution across subjects.
- **Figure 4** — mean TP matrix.

### Validations

Figure 5 (test-retest, Extension γ), Figure 6 (parcellation robustness, Extension α), and Figures 7–8 (task vs rest, Extension β) live in `results/figures/`. See `writeup/writeup.md` for the future work.

## Notes

- **Family structure** — HCP includes many twins. Jun's actual heritability and SNP analyses use a kinship matrix in mixed-effects models. This project ignores family structure (within-subject test-retest doesn't depend on it) but the limitation is flagged in `src/reliability.py` and the write-up.
- **HMM library** — code prefers `osl-dynamics` (Python port of HMM-MAR, used by Jun) and falls back to `hmmlearn` Gaussian HMM with full covariance if `osl-dynamics`' `mne` / TensorFlow runtime deps aren't installed. The fallback is a simplification — not the MAR observation model — and is flagged in `src/hmm_fit.py`.

## Repository structure

```
src/         # importable modules (data IO, HMM fit, FO/TP, plotting, extensions)
scripts/     # entry points; orchestrate, don't compute
data/        # HCP downloads (gitignored) — see data/README.md for shopping list
results/     # figures (PNG + PDF) and tables (CSV)
writeup/     # writeup.md skeleton — Lucas writes the narrative
notebooks/   # optional interactive checks
config.yaml  # paths, K values, seeds
```
