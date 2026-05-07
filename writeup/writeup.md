# HCP HMM Connectome Dynamics — Methods Portfolio
**Lucas Zhang · [date]**
**Repo:** github.com/zhangjiashu0612/hcp-hmm-dynamics

## 1. Motivation
*[1 paragraph — why dynamic FC, why HMM, framing toward Jun 2025.]*

## 2. Data and preprocessing
*[Bullet list of HCP subset, parcellations used, run structure.]*

## 3. HMM pipeline
*[K=4, group-level fit, per-subject decoding — Figures 1, 2 referenced.]*

## 4. Temporal phenotypes: FO and TP
*[Figures 3, 4 referenced; 1 paragraph on what we observe.]*

## 5. Robustness: K=6 replication
*[1 short paragraph.]*

## 6. Validation and extensions

### 6.1 Parcellation robustness (ICA-300 vs Schaefer-200)
*[Figure 6, 1 paragraph.]*

### 6.2 Task vs rest state repertoire
*[Figures 7, 8, 1 paragraph.]*

### 6.3 Test-retest reliability of FO and TP
*[Figure 5 with ICC, 1 paragraph.]*

## 7. Limitations

- **Gaussian HMM, not MAR.** The active backend is `hmmlearn.GaussianHMM` with
  full covariance — a simplified observation model relative to the multivariate-
  autoregressive HMM Jun et al. (2025) use via `osl-dynamics`. The dispatch in
  `src/hmm_fit.py` will route to `osl-dynamics` automatically once `mne` and
  TensorFlow are installed.
- **No family-structure handling.** HCP includes many twins; Jun's heritability
  analyses use a kinship matrix in mixed-effects models. Within-subject test-
  retest does not require this; group-level statistics in Extensions α and β
  would.
- **No genetics layer.** No access to dbGAP. The full Jun 2025 SNP–dynamics
  analysis is out of scope.
- **TP between-subject correlation r ≈ 0.98 on synthetic data.** Subject-level
  TPs are Dirichlet draws around a common base with mild concentration, so
  subjects share roughly the same diagonal-heavy TP. Real HCP TPs vary
  substantially more between subjects — that variability is what Jun et al.
  (2022) quantify as heritable (h² = 0.43, 95% CI [0.29, 0.57]). On real data
  this artifact resolves and the heritability framing of the test-retest
  analysis becomes empirically meaningful.
- **β rest-vs-task similarity matrix near-zero everywhere on synthetic data.**
  Task HMM was fit independently with a permuted state-bias, so independent
  random initialization yields no shared spatial structure with the rest HMM.
  On real data, rest and task share underlying neural states (Shine et al.
  2016), and the diagonal of the similarity matrix should be elevated — that
  elevation is the actual quantity of interest in Extension β.
- **Synthetic data has no true network-block structure (Figure 1 off-diagonal
  is noise).** Real ICA-300 → 139 components or any anatomically-meaningful
  parcellation will produce visible block structure aligned to canonical ICNs
  (DMN, FPN, DAN, CON, SMN, VIS, Limbic per Yeo et al. 2011), as in Jun 2022
  Figure 2A.
- **Pipeline is method-validated; data-source-dependent results are not yet
  empirical.** ICC values, FO/TP distributions, parcellation-robustness
  correlations, and rest-vs-task contrasts are all calibration artifacts of
  the synthetic generator. The real-data port (one-line swap in
  `src/data_loading.py`) is the next step.

## 8. Open questions / next steps I find compelling
*[3–4 bullet points — left for Lucas to write.]*

## References

- Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for
  evaluating normed and standardized assessment instruments in psychology.
  *Psychological Assessment*, 6(4), 284–290.
- Hansen, J. Y., Shafiei, G., Markello, R. D., Smart, K., Cox, S. M. L.,
  Nørgaard, M., et al. (2022). Mapping neurotransmitter systems to the
  structural and functional organization of the human neocortex.
  *Nature Neuroscience*, 25(11), 1569–1581.
- Jun, S., Alderson, T. H., Altmann, A., & Sadaghiani, S. (2022).
  Heritability of individualized cognitive flexibility trajectories
  in human brain dynamics. *NeuroImage*, *[volume/pages — fill in]*.
- Jun, S., et al. (2025). Modulatory neurotransmitter genotypes shape
  dynamic functional connectome reconfigurations.
  *Journal of Neuroscience*, *[volume/pages — fill in]*.
- Shine, J. M., Bissett, P. G., Bell, P. T., Koyejo, O., Balsters, J. H.,
  Gorgolewski, K. J., Moodie, C. A., & Poldrack, R. A. (2016). The dynamics
  of functional brain networks: integrated network states during cognitive
  task performance. *Neuron*, 92(2), 544–554.
- Shine, J. M., Breakspear, M., Bell, P. T., Ehgoetz Martens, K., Shine, R.,
  Koyejo, O., Sporns, O., & Poldrack, R. A. (2019). Human cognition involves
  the dynamic integration of neural activity and neuromodulatory systems.
  *Nature Neuroscience*, 22(2), 289–296.
- Yeo, B. T. T., Krienen, F. M., Sepulcre, J., Sabuncu, M. R., Lashkari, D.,
  Hollinshead, M., et al. (2011). The organization of the human cerebral
  cortex estimated by intrinsic functional connectivity. *Journal of
  Neurophysiology*, 106(3), 1125–1165.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations:
  Uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428.
- Vidaurre, D., Smith, S. M., & Woolrich, M. W. (2017). Brain network dynamics
  are hierarchically organized in time. *PNAS*, 114(48), 12827–12832.
