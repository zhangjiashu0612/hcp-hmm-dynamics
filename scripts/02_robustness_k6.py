"""K=6 robustness check on HCP rs-fMRI (intended real-data entry point).

Stub. Real-data execution gated on PTN access (see data/README.md).

Intended behavior: same loader, concatenation, and group fit as 01_fit_hmm.py
but with K=cfg["hmm"]["K_robustness"]=6. Verifies that the temporal phenotypes
(FO distribution, number of states with non-trivial occupancy >5%) replicate
under a different K, per Jun 2025's robustness check. Does not regenerate
the full extension figures; writes a CSV of K=6 FO per subject and a console
summary of state-occupancy bins.
"""

if __name__ == "__main__":
    raise NotImplementedError(
        "Real-data execution pending HCP S1200 PTN access. "
        "Pipeline is validated on synthetic data via scripts/synthetic_smoke.py."
    )
