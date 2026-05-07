"""Selectively extract HCP PTN d=300 timeseries for subjects in subject_list.txt.

Avoids the ~30 GB full extraction; pulls only `<subject>.txt` for the listed subjects
out of the 13 GB ZIP. Total extracted size: ~50-100 MB for 30 subjects.

Usage:
    python scripts/00_extract_ptn.py \\
        --zip ~/Downloads/HCP1200_Parcellation_Timeseries_Netmats.zip \\
        --subject-list data/subject_list.txt \\
        --out data/HCP/node_timeseries/3T_HCP1200_MSMAll_d300_ts2
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PTN_PREFIX = "HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d300_ts2/"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zip", required=True, type=Path, help="Path to HCP1200_Parcellation_Timeseries_Netmats.zip")
    p.add_argument("--subject-list", required=True, type=Path, help="One HCP subject ID per line")
    p.add_argument("--out", required=True, type=Path, help="Output directory for <subject>.txt files")
    args = p.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"[extract] ZIP not found: {args.zip}")
    if not args.subject_list.exists():
        raise SystemExit(f"[extract] Subject list not found: {args.subject_list}")

    subjects = {s.strip() for s in args.subject_list.read_text().splitlines() if s.strip() and not s.startswith("#")}
    if not subjects:
        raise SystemExit(f"[extract] No subject IDs in {args.subject_list}")
    print(f"[extract] {len(subjects)} subjects requested")

    args.out.mkdir(parents=True, exist_ok=True)

    found, missing = [], set(subjects)
    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            if not info.filename.startswith(PTN_PREFIX) or info.is_dir():
                continue
            stem = Path(info.filename).stem
            if stem in subjects:
                target = args.out / f"{stem}.txt"
                with zf.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())
                found.append(stem)
                missing.discard(stem)
                print(f"[extract]   {stem}.txt  ({info.file_size / 1e6:.1f} MB)")

    print(f"[extract] Extracted {len(found)} files to {args.out}")
    if missing:
        print(f"[extract] WARNING: {len(missing)} subjects not found in ZIP: {sorted(missing)}")


if __name__ == "__main__":
    main()
