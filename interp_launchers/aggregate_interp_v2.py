#!/usr/bin/env python3
"""Aggregate per-slide interp CSVs into a cohort table + a clean feature matrix.

Run after the run_*_interp.sh array jobs finish.

  python aggregate_interp_v2.py --stain hne
  python aggregate_interp_v2.py --stain trichrome

Writes:
  output/interp_v2/<stain>/<stain>_features.csv     full table (one row per sample)
  output/feature_matrices/interp_<key>_v2.csv       sample_id + feature columns only
  output/feature_matrices/interp_<key>_v2.meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_CBL = Path("/scratch/st-singha53-1/bhelfert/CBL")
MAT = _CBL / "output" / "feature_matrices"

CFG = {
    "hne":       dict(dir=_CBL / "output/interp_v2/hne",       name="he_features",       key="interp_hne_v2"),
    "trichrome": dict(dir=_CBL / "output/interp_v2/trichrome", name="trichrome_features", key="interp_tri_v2"),
}
ID_COLS = ["slide_id", "sample_id", "vsr", "outcome", "subset", "pulm_fib", "normalized"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stain", required=True, choices=list(CFG))
    args = ap.parse_args()
    cfg = CFG[args.stain]
    per = cfg["dir"] / "per_slide"
    files = sorted(per.glob("*.csv"))
    files = [f for f in files if not f.stem.endswith("_normalized")]
    if not files:
        raise SystemExit(f"no per-slide CSVs in {per}")
    print(f"[{args.stain}] {len(files)} per-slide CSVs")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values("sample_id").reset_index(drop=True)
    cohort_csv = cfg["dir"] / f"{cfg['name']}.csv"
    df.to_csv(cohort_csv, index=False)
    print(f"  cohort table -> {cohort_csv}  ({len(df)} samples)")

    feat_cols = [c for c in df.columns if c not in ID_COLS]
    mat = df[["sample_id"] + feat_cols].copy()
    # drop any all-NaN feature columns; report partial-NaN
    nan_counts = mat[feat_cols].isna().sum()
    drop = nan_counts[nan_counts == len(mat)].index.tolist()
    if drop:
        print(f"  dropping {len(drop)} all-NaN feature cols: {drop}")
        mat = mat.drop(columns=drop)
    partial = nan_counts[(nan_counts > 0) & (nan_counts < len(mat))]
    if len(partial):
        print(f"  partial-NaN cols (left for median-impute downstream): {dict(partial)}")
    out = MAT / f"{cfg['key']}.csv"
    mat.to_csv(out, index=False)
    meta = {"key": cfg["key"], "stain": args.stain, "n_samples": int(len(mat)),
            "n_features": int(len([c for c in mat.columns if c != "sample_id"])),
            "source": str(cohort_csv),
            "assignment": "TRIDENT contour polygon-cluster + debris drop",
            "normalization": "none (raw)"}
    (MAT / f"{cfg['key']}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  matrix -> {out}  n={len(mat)} feats={meta['n_features']}")
    print(f"  samples: {sorted(mat['sample_id'])}")


if __name__ == "__main__":
    main()
