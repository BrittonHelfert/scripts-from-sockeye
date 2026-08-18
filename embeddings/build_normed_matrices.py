#!/usr/bin/env python3
"""Build per-sample feature matrices from the new cohort-median-normed embeddings.

For every slide:
  1. assign patches to samples by TRIDENT-contour CONTAINMENT (the corrected
     method from common.contour_assign -- handles the vsr344/vsr346 x-overlap and
     drops debris/off-tissue patches), then
  2. pool the patch embeddings per sample -> one row per sample, using one of:
       --pool l2   : L2-NORMALIZE each patch embedding (unit length) before
                     mean-pooling (the "_mnorm" matrices). Weights every patch
                     equally regardless of raw magnitude, so a few high-norm
                     patches can't dominate the sample mean.
       --pool mean : plain mean-pool of the raw patch embeddings, no per-patch
                     normalization (the "_mmean" matrices).

Both modes read the SAME cohort-median stain-normalized embeddings; the only
difference is whether each patch is L2-normalized before averaging. Output keys
swap the suffix accordingly: <enc>_<stain>_mnorm (l2) vs <enc>_<stain>_mmean (mean).

Reads the Stage-2 outputs (features_<enc>_vahadane_med / features_<enc>_reinhardmed).
Writes feature_matrices/<key>.csv + <key>.meta.json.  CPU only.

  python build_normed_matrices.py                       # all four sets, L2-pooled (_mnorm)
  python build_normed_matrices.py --pool mean           # all four sets, mean-pooled (_mmean)
  python build_normed_matrices.py --sets resnet_hne_mnorm --pool mean
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_PREPROC = Path(__file__).resolve().parent
sys.path.insert(0, str(_PREPROC))
from common import contour_assign  # noqa: E402

_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_CBL = Path("/scratch/st-singha53-1/bhelfert/CBL")
FEAT = _CBL / "output" / "features"
MAT = _CBL / "output" / "feature_matrices"
MAP_CSV = _DATA / "he_tri_image_mapping.csv"
DEBRIS_FRAC = 0.05

# key -> features dir, contours dir, id column in the mapping CSV
SETS = {
    "resnet_hne_mnorm": dict(
        feat=FEAT / "resnet50_hne_20x/20x_256px_0px_overlap/features_resnet50_vahadane_med",
        contours=FEAT / "resnet50_hne_20x/contours_geojson", id_col="he"),
    "uni_hne_mnorm": dict(
        feat=FEAT / "uni_v1_hne/20x_256px_0px_overlap/features_uni_v1_vahadane_med",
        contours=FEAT / "uni_v1_hne/contours_geojson", id_col="he"),
    "resnet_tri_mnorm": dict(
        feat=FEAT / "resnet50_trichrome_20x/20x_256px_0px_overlap/features_resnet50_reinhardmed",
        contours=FEAT / "resnet50_trichrome_20x/contours_geojson", id_col="trichrome"),
    "conch_tri_mnorm": dict(
        feat=FEAT / "conch_v1_trichrome/20x_512px_0px_overlap/features_conch_v1_reinhardmed",
        contours=FEAT / "conch_v1_trichrome/contours_geojson", id_col="trichrome"),
}


def load_mapping():
    df = pd.read_csv(MAP_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def vsr_ids_for(row):
    return [str(row[c]).strip() for c in ("sub1", "sub2", "sub3", "sub4")
            if pd.notna(row[c]) and str(row[c]).strip().upper() != "NA"]


def l2_pool(feats):
    """Unit-normalize each patch (row) then mean-pool."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (feats / norms).mean(axis=0)


def mean_pool(feats):
    """Plain mean-pool of the raw patch embeddings (no per-patch normalization)."""
    return feats.mean(axis=0)


# pool mode -> (pooling fn, output-key suffix, meta "pooling" description)
POOLERS = {
    "l2":   (l2_pool,   "_mnorm", "per-patch L2 normalize, then mean-pool"),
    "mean": (mean_pool, "_mmean", "plain mean-pool (no per-patch normalization)"),
}


def build_set(key, mapping, pool="l2"):
    cfg = SETS[key]
    fdir, cdir, id_col = cfg["feat"], cfg["contours"], cfg["id_col"]
    pool_fn, suffix, pool_desc = POOLERS[pool]
    out_key = key.replace("_mnorm", suffix)
    if not fdir.exists():
        print(f"[{out_key}] features dir missing ({fdir}) — SKIP")
        return None
    rows, report = {}, []
    for _, mrow in mapping.iterrows():
        slide = mrow[id_col]
        if pd.isna(slide):
            continue
        slide = int(slide)
        h5 = fdir / f"{slide}.h5"
        gj = cdir / f"{slide}.geojson"
        vsr_ids = vsr_ids_for(mrow)
        if not h5.exists():
            report.append(f"  slide {slide}: no h5 — {vsr_ids} skipped")
            continue
        if not gj.exists():
            report.append(f"  slide {slide}: no geojson — {vsr_ids} skipped")
            continue
        with h5py.File(h5, "r") as f:
            coords = f["coords"][:]
            patch_l0 = float(f["coords"].attrs["patch_size_level0"])
            feats = f["features"][:]
        assign, _, info = contour_assign.assign_patches(
            coords, patch_l0, gj, vsr_ids, debris_frac=DEBRIS_FRAC)
        if info["status"] != "ok":
            report.append(f"  slide {slide}: assign {info['status']} — {vsr_ids} skipped")
            continue
        for vsr, ix in assign.items():
            if ix.size == 0:
                report.append(f"  slide {slide} {vsr}: 0 patches — no row")
                continue
            rows[vsr] = pool_fn(feats[ix])
        report.append(f"  slide {slide}: {info['n_patches']} patches, "
                      f"{info['n_offtissue_dropped']} off-tissue dropped, "
                      f"per-sample={info['n_per_sample']}")

    if not rows:
        print(f"[{out_key}] produced no rows — SKIP")
        return None
    dim = len(next(iter(rows.values())))
    df = pd.DataFrame.from_dict(rows, orient="index",
                                columns=[f"f{i}" for i in range(dim)])
    df.index.name = "sample_id"
    df = df.sort_index().reset_index()
    out = MAT / f"{out_key}.csv"
    df.to_csv(out, index=False)
    meta = {"key": out_key, "n_samples": int(len(df)), "dim": dim,
            "features_dir": str(fdir), "contours_dir": str(cdir),
            "assignment": "contour_containment (common.contour_assign)",
            "debris_frac": DEBRIS_FRAC,
            "pooling": pool_desc,
            "normalization": "cohort-median Vahadane (HNE) / Reinhard (trichrome)"}
    (MAT / f"{out_key}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{out_key}] wrote {out.name}  n={len(df)} dim={dim}")
    for line in report:
        print(line)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=list(SETS),
                    choices=list(SETS))
    ap.add_argument("--pool", choices=list(POOLERS), default="l2",
                    help="l2 = per-patch L2 then mean (_mnorm); "
                         "mean = plain mean-pool (_mmean). Default l2.")
    args = ap.parse_args()
    mapping = load_mapping()
    for key in args.sets:
        print(f"\n=== {key} (--pool {args.pool}) ===")
        build_set(key, mapping, pool=args.pool)
    print("\nDone.")


if __name__ == "__main__":
    main()
