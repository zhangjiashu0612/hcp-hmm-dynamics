"""HCP timeseries IO + config loader.

Primary loader handles the PTN release format: per-subject .txt with the four
HCP rs-fMRI runs concatenated along the time axis, shape (4 * 1200, n_components).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# HCP rs-fMRI structure: 4 runs of 1200 timepoints each, TR = 0.72 s.
RUN_LENGTH = 1200
N_RUNS = 4


def load_config(path: str | Path = "config.yaml") -> dict:
    """Read the project config.yaml. All scripts go through here."""
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_subject_list(cfg: dict, ica300_dir: Path) -> list[str]:
    """Use cfg['paths']['subject_list'] if it exists; otherwise first 30 alphabetical."""
    listed = Path(cfg["paths"]["subject_list"])
    if listed.exists():
        ids = [s.strip() for s in listed.read_text().splitlines() if s.strip() and not s.startswith("#")]
        return ids[: cfg["n_subjects"]]
    print(f"[data] WARNING: {listed} missing — picking first {cfg['n_subjects']} alphabetical subjects "
          f"from {ica300_dir} without enforcing family structure.")
    files = sorted(ica300_dir.glob("*.txt"))
    return [f.stem for f in files[: cfg["n_subjects"]]]


def load_ica300_subject(subject_id: str, ica300_dir: Path) -> np.ndarray:
    """Load HCP group-ICA d=300 timeseries for one subject. Returns (T, 300)."""
    path = ica300_dir / f"{subject_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"[data] Missing PTN file: {path}")
    return np.loadtxt(path)


def split_runs(ts: np.ndarray, n_runs: int = N_RUNS, run_length: int = RUN_LENGTH) -> list[np.ndarray]:
    """Split concatenated subject timeseries into per-run blocks (HCP PTN format)."""
    expected = n_runs * run_length
    if ts.shape[0] != expected:
        # Some PTN copies have slightly different lengths — pass through best-effort.
        print(f"[data] WARNING: timeseries length {ts.shape[0]} != expected {expected}; "
              f"splitting into {n_runs} equal chunks.")
        run_length = ts.shape[0] // n_runs
    return [ts[i * run_length : (i + 1) * run_length] for i in range(n_runs)]


def apply_components_mask(ts: np.ndarray, mask_idx: np.ndarray) -> np.ndarray:
    """Subset components by integer index (e.g., 139 cortical/subcortical out of 300)."""
    return ts[:, mask_idx]


def load_components_mask(cfg: dict) -> np.ndarray | None:
    """Optional cortical/subcortical-only mask. Returns None if not provided."""
    p = Path(cfg["paths"]["components_mask"])
    if not p.exists():
        return None
    return np.loadtxt(p, dtype=int)


def standardize_timeseries(ts: np.ndarray) -> np.ndarray:
    """Per-component z-score (zero mean, unit variance) along the time axis."""
    return (ts - ts.mean(axis=0, keepdims=True)) / (ts.std(axis=0, keepdims=True) + 1e-12)
