#!/usr/bin/env python3
"""
Export resnet_hne_norm and uni_hne_norm feature matrices from vahadane-normalized
WSI h5 files. Splits each WSI into per-sample tissue pieces using x-coordinate
gaps (matching _cluster_patches_by_x_gaps in investigation/replication_cv.py).

Usage:
    python export_vahadane_matrices.py
"""
import os
import glob as gl
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
FEAT_BASE = Path("/scratch/st-singha53-1/bhelfert/CBL/output/features")
OUTDIR = Path("/scratch/st-singha53-1/bhelfert/CBL/output/feature_matrices")
SLIDE_MAP_CSV = "/arc/project/st-singha53-1/datasets/ssc/he_tri_image_mapping.csv"
EXCLUDE = {"vsr346"}

FEATURE_SETS = {
    "resnet_hne_norm": {"dir": FEAT_BASE / "resnet50_hne_20x/20x_256px_0px_overlap/features_resnet50_vahadane", "stain_col": "he"},
    "uni_hne_norm":    {"dir": FEAT_BASE / "uni_v1_hne/20x_256px_0px_overlap/features_uni_v1_vahadane",        "stain_col": "he"},
    "resnet_tri":      {"dir": FEAT_BASE / "resnet50_trichrome_20x/20x_256px_0px_overlap/features_resnet50",   "stain_col": "trichrome"},
    "conch_tri_norm":  {"dir": FEAT_BASE / "conch_v1_trichrome/20x_512px_0px_overlap/features_conch_v1_reinhard",   "stain_col": "trichrome"},
    "resnet_tri_norm": {"dir": FEAT_BASE / "resnet50_trichrome_20x/20x_256px_0px_overlap/features_resnet50_reinhard", "stain_col": "trichrome"},
}


# ── Tissue splitting ──────────────────────────────────────────────────────────

def _cluster_patches_by_x_gaps(coords, n_pieces):
    """Split patches into tissue pieces using x-coordinate gaps."""
    x = coords[:, 0]
    sort_idx = np.argsort(x)
    if n_pieces <= 1:
        return [np.arange(len(coords))]
    gaps = np.diff(x[sort_idx])
    gap_positions = np.sort(np.argsort(gaps)[::-1][:n_pieces - 1])
    groups = []
    prev = 0
    for gp in gap_positions:
        groups.append(sort_idx[prev:gp + 1])
        prev = gp + 1
    groups.append(sort_idx[prev:])
    groups.sort(key=lambda g: coords[g, 0].mean())
    return groups


# ── Loader ────────────────────────────────────────────────────────────────────

def load_wsi_features(feat_dir, stain_col="he"):
    """Load WSI-level h5 files, split to samples, mean-pool patches."""
    h5_paths = gl.glob(str(feat_dir / "*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No H5 files in {feat_dir}")

    # Group h5 paths by slide_id
    h5_by_slide = {}
    for p in h5_paths:
        basename = os.path.splitext(os.path.basename(p))[0]
        slide_id = int(basename.split("_tissue_")[0]) if "_tissue_" in basename else int(basename)
        h5_by_slide.setdefault(slide_id, []).append(p)
    for k in h5_by_slide:
        h5_by_slide[k].sort()

    mapping = pd.read_csv(SLIDE_MAP_CSV)
    records = {}

    for _, row in mapping.iterrows():
        slide_id = int(row[stain_col])
        patient_ids = [
            str(row[c]).strip()
            for c in ["sub1", "sub2", "sub3", "sub4"]
            if pd.notna(row[c]) and str(row[c]).strip() != "NA"
        ]
        if slide_id not in h5_by_slide:
            continue

        h5_list = h5_by_slide[slide_id]
        n_pieces = len(patient_ids)

        if len(h5_list) > 1 or "_tissue_" in os.path.basename(h5_list[0]):
            # Already split into per-piece h5 files
            if len(h5_list) != n_pieces:
                print(f"  SKIP slide {slide_id}: {len(h5_list)} files != {n_pieces} patients")
                continue
            for pid, hp in zip(patient_ids, h5_list):
                if pid in EXCLUDE:
                    continue
                with h5py.File(hp, "r") as f:
                    records[pid] = f["features"][:].mean(axis=0)
        else:
            # Single h5 per slide — split by x-coordinate gaps
            with h5py.File(h5_list[0], "r") as f:
                coords = f["coords"][:]
                feats = f["features"][:]
            groups = _cluster_patches_by_x_gaps(coords, n_pieces)
            if len(groups) != n_pieces:
                print(f"  SKIP slide {slide_id}: got {len(groups)} groups != {n_pieces} patients")
                continue
            for pid, idx in zip(patient_ids, groups):
                if pid in EXCLUDE:
                    continue
                records[pid] = feats[idx].mean(axis=0)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in FEATURE_SETS.items():
        feat_dir  = cfg["dir"]
        stain_col = cfg["stain_col"]
        print(f"\n{'='*60}")
        print(f"Exporting: {name}")
        print(f"  Source: {feat_dir}")

        records = load_wsi_features(feat_dir, stain_col=stain_col)
        if not records:
            print("  SKIPPED: no features loaded")
            continue

        sample_ids = sorted(records.keys())
        X = np.vstack([records[s] for s in sample_ids])
        dim = X.shape[1]
        columns = [f"f{i}" for i in range(dim)]

        df = pd.DataFrame(X, columns=columns)
        df.insert(0, "sample_id", sample_ids)

        out_path = OUTDIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Shape: {len(sample_ids)} samples x {dim} features")

    print(f"\nDone. Matrices saved to: {OUTDIR}")


if __name__ == "__main__":
    main()
