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

**Primary source — ConnectomeDB:**
- Sign in at [db.humanconnectome.org](https://db.humanconnectome.org)
- Navigate: HCP_Resources project → GroupAvg → `HCP1200_Parcellation_Timeseries_Netmats.zip`
- Direct URL: <https://db.humanconnectome.org/app/action/ChooseDownloadResources?project=HCP_Resources&resource=GroupAvg&filePath=HCP1200_Parcellation_Timeseries_Netmats.zip>

**Alternate source — BALSA (same file, mirror):**
- [balsa.wustl.edu](https://balsa.wustl.edu) → "ConnectomeDB powered by BALSA"
- HCP-Young Adult 2025 → **Files** tab → S1200 archived resources → `HCP1200_Parcellation_Timeseries_Netmats.zip`

Save the ZIP wherever you have ~13 GB free. Below assumes `~/Downloads/HCP1200_Parcellation_Timeseries_Netmats.zip`.

### Step 2 — write your subject list

Put one HCP subject ID per line in `data/subject_list.txt` (e.g., `100307`, `100408`, ...). This must exist **before** the selective-extraction step. Target 30 unrelated subjects; if the file is missing the pipeline picks the first 30 alphabetically and prints a warning that family structure was not enforced.

### Step 3 — extract only what we need (recommended: Python helper)

The ZIP contains d=15/25/50/100/200/300 timeseries plus netmats; we only want d=300 for our 30 subjects.

**Recommended — Python selective extraction (subject-list-aware):**
```bash
conda activate hcp-hmm
python scripts/00_extract_ptn.py \
    --zip ~/Downloads/HCP1200_Parcellation_Timeseries_Netmats.zip \
    --subject-list data/subject_list.txt \
    --out data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2
```
This reads `subject_list.txt`, walks the ZIP central directory, and extracts only `<subject>.txt` files matching `3T_HCP1200_MSMAll_d300_ts2/`. Total extracted size: ~50–100 MB for 30 subjects.

**Alternative — one-liner unzip (extracts all PTN subjects' d=300, ~1 GB; you can prune after):**
```bash
unzip -j ~/Downloads/HCP1200_Parcellation_Timeseries_Netmats.zip \
    "HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d300_ts2/*.txt" \
    -d data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2/
# Then prune to subject_list.txt:
cd data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2
ls *.txt | grep -vFf <(awk '{print $1".txt"}' ../../../subject_list.txt) | xargs -r rm
# Optional: delete the ZIP once the extraction succeeds
rm ~/Downloads/HCP1200_Parcellation_Timeseries_Netmats.zip
```

### Expected on-disk layout

```
data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2/
├── 100307.txt                    # 4 runs concatenated, shape ≈ (4*1200, 300)
├── 100408.txt
├── 100610.txt
└── ...                           # one file per subject ID in subject_list.txt
```

The HCP PTN release ships the four runs **already concatenated along the time axis** in this single per-subject file. The pipeline assumes that and splits back into 4 runs of 1200 timepoints each (TR = 0.72 s). If your copy is split per-run instead (`<subject>_REST1_LR.txt` etc.), tell me and I'll add a loader branch.

### Optional — 139-component mask

`data/139_components_mask.txt`, one zero-indexed component per line, listing the cortical/subcortical components to keep (excluding cerebellum / brainstem). If absent, all 300 components are used and a warning is emitted.

---

## Extension α — Schaefer-200 source data

The Schaefer-200 atlas is a **cortical** parcellation; we need the underlying voxel/grayordinate data to parcellate ourselves. These files **are** on the `hcp-openaccess` S3 bucket.

**Preferred (grayordinate CIFTI):**
```
data/HCP/cifti/<subject>/MNINonLinear/Results/rfMRI_<RUN>/rfMRI_<RUN>_Atlas_MSMAll_hp2000_clean.dtseries.nii
# RUN ∈ {REST1_LR, REST1_RL, REST2_LR, REST2_RL}
```
Same minimal preprocessing + ICA-FIX (`hp2000_clean`) as the d=300 release. Schaefer ships in fsLR space so we can sample directly from grayordinate dtseries.

```bash
# Per-subject CIFTI download (example for subject 100307):
aws s3 cp --recursive --no-sign-request \
    s3://hcp-openaccess/HCP_1200/100307/MNINonLinear/Results/rfMRI_REST1_LR/ \
    data/HCP/cifti/100307/MNINonLinear/Results/rfMRI_REST1_LR/ \
    --exclude "*" --include "rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii"
```

**Fallback (volumetric MNI):**
```
data/HCP/volumetric/<subject>/MNINonLinear/Results/rfMRI_<RUN>/rfMRI_<RUN>_hp2000_clean.nii.gz
```
We'll resample Schaefer-200 atlas to MNI 2 mm via `nilearn.datasets.fetch_atlas_schaefer_2018` and apply with `NiftiLabelsMasker`. This introduces a mild grayordinate-vs-volumetric mismatch with the d=300 ICA which we'll flag in the write-up.

You only need **one of the two** for the same 30 subjects used in Layer 1. Tell me which is feasible at Checkpoint 3.

---

## Extension β — task fMRI

Two tasks: **WM** (working memory) and **LANGUAGE**. Two phase-encoding directions per task (LR and RL).

**Important — task d=300 timeseries:** the HCP PTN release ships **rest-only** at d=300; there is no task PTN counterpart. To reuse the d=300 spatial maps for task data we need to project task CIFTI/volumetric data through the group-ICA spatial maps ourselves (an extra step in the pipeline; doable). Lucas — please confirm at Checkpoint 3 whether you've located task d=300 timeseries from any source; otherwise plan for the projection step.

**CIFTI task data (S3):**
```
data/HCP/task/cifti/<subject>/MNINonLinear/Results/tfMRI_<TASK>_<DIR>/tfMRI_<TASK>_<DIR>_Atlas_MSMAll.dtseries.nii
# TASK ∈ {WM, LANGUAGE}, DIR ∈ {LR, RL}
```

```bash
# Example — pull WM_LR CIFTI for one subject:
aws s3 cp --no-sign-request \
    s3://hcp-openaccess/HCP_1200/100307/MNINonLinear/Results/tfMRI_WM_LR/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii \
    data/HCP/task/cifti/100307/MNINonLinear/Results/tfMRI_WM_LR/
```

Subject count: same 30 as Layer 1 by default; can reduce to 5–10 if bandwidth is tight.

---

## Sanity checks before running anything

After extraction, confirm shape with:

```bash
conda activate hcp-hmm
python -c "
import numpy as np, pathlib
p = pathlib.Path('data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2')
for f in sorted(p.glob('*.txt'))[:3]:
    a = np.loadtxt(f)
    print(f.name, a.shape)
"
```

Expected output for each: `<subject>.txt (4800, 300)` (4 runs × 1200 TRs × 300 components).

---

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
