# HCP data list

## Status (May 2026)

The HCP S1200 PTN package (`HCP1200_Parcellation_Timeseries_Netmats.zip`, ~13 GB) is currently inaccessible. The legacy ConnectomeDB endpoint redirects to BALSA without serving the file; the new BALSA HCP-Young Adult 2025 release does not redistribute the d=300 ICA package. The pipeline is currently validated on synthetic data via `scripts/synthetic_smoke.py`; this file documents the intended download path for when access is restored.

---

> **Provenance.** S3 paths verified against AWS Open Data Registry; PTN ZIP path verified via HCP-Users mailing list archive [Elam 2018].

Everything in `data/` (except this file, the dotfiles, `subject_list.txt`, and the optional `139_components_mask.txt`) is gitignored. 

The pipeline reads paths from [`../config.yaml`](../config.yaml) — adjust the `paths:` section if you put data somewhere else.

---

## TL;DR — what to grab

| Layer | What | Source | Priority |
| --- | --- | --- | --- |
| Layer 1 (primary) | HCP-provided **group-ICA d=300** parcellated rs-fMRI timeseries for ≥30 unrelated subjects | ConnectomeDB / BALSA — single ~13 GB ZIP | **REQUIRED to start** |
| Extension γ | Same as Layer 1 — 4 runs per subject, no extra download | — | comes free |
| Extension α | Either dense CIFTI rs-fMRI (preferred) or volumetric MNI rs-fMRI, same subjects | HCP S1200 minimally preprocessed release on S3 | **needed before α** |
| Extension β | Task fMRI (`WM` and `LANGUAGE`) — see notes below | HCP S1200 task release | **needed before β** |

Subject count: target 30 unrelated subjects for Layer 1 / γ / α. Extension β can drop to 5–10 if bandwidth is tight.

**Disk budget:** ZIP is ~13 GB and full extraction is ~30 GB → peak ~43 GB. The recommended workflow below extracts only the d=300 files for subjects in `subject_list.txt`, dropping peak usage to ZIP + a few hundred MB.

---

## Layer 1 — group-ICA d=300 timeseries (PRIMARY)

The HCP PTN (Parcellated Timeseries and Netmats) release is distributed as a **single ZIP file** from ConnectomeDB. The `hcp-openaccess` S3 bucket only contains per-subject raw CIFTI/NIfTI; it does **not** contain the group-ICA parcellated timeseries.

### Step 1 — download the ZIP (~13 GB)

### Step 2 — write your subject list

Put one HCP subject ID per line in `data/subject_list.txt` (e.g., `100307`, `100408`, ...). This must exist **before** the selective-extraction step. Target 30 unrelated subjects; if the file is missing the pipeline picks the first 30 alphabetically and prints a warning that family structure was not enforced.

### Step 3 — extract only what we need 

The ZIP contains d=15/25/50/100/200/300 timeseries plus netmats; this project only want d=300 for our 30 subjects.

## Extension α — Schaefer-200 source data

The Schaefer-200 atlas is a **cortical** parcellation; we need the underlying voxel/grayordinate data to parcellate ourselves. These files **are** on the `hcp-openaccess` S3 bucket.

We'll resample Schaefer-200 atlas to MNI 2 mm via `nilearn.datasets.fetch_atlas_schaefer_2018` and apply with `NiftiLabelsMasker`. This introduces a mild grayordinate-vs-volumetric mismatch with the d=300 ICA.

## Extension β — task fMRI

Two tasks: **WM** (working memory) and **LANGUAGE**. Two phase-encoding directions per task (LR and RL).

**Important — task d=300 timeseries:** the HCP PTN release ships **rest-only** at d=300; there is no task PTN counterpart. To reuse the d=300 spatial maps for task data we need to project task CIFTI/volumetric data through the group-ICA spatial maps ourselves (an extra step in the pipeline; doable). So not done for now.

## What you do NOT need to download

- **dbGAP genetics** — not used in this project.
- **Behavioral / cognitive battery CSVs** — not used.
- **Structural MRI / DWI** — not used.
- **Non-MSMAll versions** of the parcellated timeseries — Jun uses MSMAll; we follow that.
- **HCP S1200 raw rs-fMRI** for Layer 1 / γ — the PTN ZIP is sufficient; only Extensions α and β need raw CIFTI/volumetric.

---

## Reference

- Elam, J. S. (2018). HCP-Users mailing list — clarification on PTN release distribution. <https://www.mail-archive.com/hcp-users@humanconnectome.org/>.
- AWS Open Data Registry — Human Connectome Project. <https://registry.opendata.aws/hcp-openaccess/>.
