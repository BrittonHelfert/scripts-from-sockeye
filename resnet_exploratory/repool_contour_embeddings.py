#!/usr/bin/env python3
"""Re-pool flagged slides' embedding rows using TRIDENT-contour containment.

The default pipeline assigns embedding patches to samples by a 1-D patch x-gap
split. On a handful of slides that mis-assigns:
  - debris flecks swept into a neighbour (vsr292/331/339 tri, vsr286/307/328/344 hne)
  - two real pieces overlapping in x mixed together (vsr344<->vsr346 on tri 269311)

This re-pools ONLY the flagged slides (all samples on them) via
common.contour_assign.assign_patches (contour containment + debris drop), and
overwrites those samples' rows in the mean-pooled feature matrices. The original
matrices are backed up to output/feature_matrices/_pre_contour_fix/ first.

Non-flagged slides are left untouched (x-gap and containment agree there).

Run on a compute node (CPU only). See run_repool_contour_embeddings.sh.
"""
from __future__ import annotations

import shutil
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
BACKUP = MAT / "_pre_contour_fix"
MAP_CSV = _DATA / "he_tri_image_mapping.csv"
DEBRIS_FRAC = 0.05

# Flagged slides per stain (slide id = the stain's mapping column).
FLAGGED = {"hne": [261212, 261230, 261262, 261317],
           "trichrome": [269299, 269308, 269310, 269311]}

# One entry per TRIDENT root; raw + normalised encoders share the patch grid +
# contours, so containment is computed once per (root, slide) and reused.
ROOTS = {
    "hne": [
        {"root": FEAT / "uni_v1_hne", "patch_subdir": "20x_256px_0px_overlap",
         "matrices": [("uni_hne", "features_uni_v1"),
                      ("uni_hne_norm", "features_uni_v1_vahadane")]},
        {"root": FEAT / "resnet50_hne_20x", "patch_subdir": "20x_256px_0px_overlap",
         "matrices": [("resnet_hne", "features_resnet50"),
                      ("resnet_hne_norm", "features_resnet50_vahadane")]},
    ],
    "trichrome": [
        {"root": FEAT / "conch_v1_trichrome", "patch_subdir": "20x_512px_0px_overlap",
         "matrices": [("conch_tri", "features_conch_v1"),
                      ("conch_tri_norm", "features_conch_v1_reinhard")]},
        {"root": FEAT / "resnet50_trichrome_20x", "patch_subdir": "20x_256px_0px_overlap",
         "matrices": [("resnet_tri", "features_resnet50"),
                      ("resnet_tri_norm", "features_resnet50_reinhard")]},
    ],
}


def load_mapping():
    df = pd.read_csv(MAP_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def vsr_ids_for(row):
    return [str(row[c]).strip() for c in ("sub1", "sub2", "sub3", "sub4")
            if pd.notna(row[c]) and str(row[c]).strip().upper() != "NA"]


def backup_matrix(key):
    src = MAT / f"{key}.csv"
    if not src.exists():
        return False
    BACKUP.mkdir(parents=True, exist_ok=True)
    dst = BACKUP / f"{key}.csv"
    if not dst.exists():               # back up the pristine version only once
        shutil.copy2(src, dst)
    return True


def main(dry_run=False):
    mapping = load_mapping()
    for stain, specs in ROOTS.items():
        id_col = "he" if stain == "hne" else "trichrome"
        slides = FLAGGED[stain]
        print(f"\n=== {stain}  slides={slides} ===")
        for spec in specs:
            root, patch_subdir = spec["root"], spec["patch_subdir"]
            contours_dir = root / "contours_geojson"
            enc0 = spec["matrices"][0][1]
            for slide_id in slides:
                row = mapping[mapping[id_col].astype("Int64") == slide_id].iloc[0]
                vsr_ids = vsr_ids_for(row)
                h5_ref = root / patch_subdir / enc0 / f"{slide_id}.h5"
                gj = contours_dir / f"{slide_id}.geojson"
                if not h5_ref.exists() or not gj.exists():
                    print(f"  [{root.name}] slide {slide_id}: missing h5/geojson — skip")
                    continue
                with h5py.File(h5_ref, "r") as f:
                    coords = f["coords"][:]
                    patch_l0 = float(f["coords"].attrs["patch_size_level0"])
                assign, kept, info = contour_assign.assign_patches(
                    coords, patch_l0, gj, vsr_ids, debris_frac=DEBRIS_FRAC)
                if info["status"] != "ok":
                    print(f"  [{root.name}] slide {slide_id}: {info['status']} "
                          f"(polys={info.get('n_polys')}) — SKIP, left as x-gap")
                    continue
                print(f"  [{root.name}] slide {slide_id} {vsr_ids}: "
                      f"{info['n_patches']} patches, "
                      f"{info['n_offtissue_dropped']} off-tissue patches dropped, "
                      f"debris polys={info['dropped_debris']}, "
                      f"kept/sample={info['n_per_sample']}")

                # Re-pool every encoder under this root with the same indices.
                for key, enc in spec["matrices"]:
                    h5 = root / patch_subdir / enc / f"{slide_id}.h5"
                    mat_path = MAT / f"{key}.csv"
                    if not h5.exists() or not mat_path.exists():
                        print(f"      {key}: missing h5/matrix — skip")
                        continue
                    with h5py.File(h5, "r") as f:
                        feats = f["features"][:]
                    df = pd.read_csv(mat_path)
                    fcols = [c for c in df.columns if c != "sample_id"]
                    n_changed = 0
                    for vsr, idx in assign.items():
                        if idx.size == 0:
                            print(f"      {key}: {vsr} has 0 patches after fix — "
                                  f"leaving old row")
                            continue
                        pooled = feats[idx].mean(axis=0)
                        m = df["sample_id"] == vsr
                        if not m.any():
                            continue
                        if not dry_run:
                            df.loc[m, fcols] = pooled
                        n_changed += 1
                    if not dry_run:
                        backup_matrix(key)
                        df.to_csv(mat_path, index=False)
                    print(f"      {key}: {'(dry) ' if dry_run else ''}"
                          f"updated {n_changed} sample rows")
    print("\nDone." + ("  [DRY RUN — no files written]" if dry_run else
                        f"  Backups in {BACKUP}"))


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
