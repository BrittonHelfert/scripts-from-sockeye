#!/usr/bin/env python3
"""Drop debris polygons from the TRIDENT contour GeoJSONs of the flagged slides.

Mirrors the debris rule used when re-pooling embeddings
(common.contour_assign): cluster a slide's polygons into n_samples groups by
x-gap, flag any polygon whose area < debris_frac * (largest polygon in its
cluster), and remove those features from the GeoJSON. This keeps the contours
consistent with the corrected per-sample feature rows (and with any future
interp re-run, which reads these same GeoJSONs).

Operates at GeoJSON-feature granularity (each TRIDENT feature is one tissue
polygon). Originals are backed up to
{contours_geojson}/_pre_debris_drop/{slide}.geojson before rewriting.

Run on a compute node (CPU only).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape

_PREPROC = Path(__file__).resolve().parent
sys.path.insert(0, str(_PREPROC))
from common import contour_assign  # noqa: E402

_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_CBL = Path("/scratch/st-singha53-1/bhelfert/CBL")
FEAT = _CBL / "output" / "features"
MAP_CSV = _DATA / "he_tri_image_mapping.csv"
DEBRIS_FRAC = 0.05

# Same flagged slides + roots as the embedding re-pool.
FLAGGED = {"hne": [261212, 261230, 261262, 261317],
           "trichrome": [269299, 269308, 269310, 269311]}
ROOTS = {
    "hne": [FEAT / "uni_v1_hne", FEAT / "resnet50_hne_20x"],
    "trichrome": [FEAT / "conch_v1_trichrome", FEAT / "resnet50_trichrome_20x"],
}


def load_mapping():
    df = pd.read_csv(MAP_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def vsr_ids_for(row):
    return [str(row[c]).strip() for c in ("sub1", "sub2", "sub3", "sub4")
            if pd.notna(row[c]) and str(row[c]).strip().upper() != "NA"]


def debris_feature_indices(gj, n_samples, debris_frac):
    """Indices of GeoJSON features that are debris (per-cluster area rule)."""
    feats = gj["features"]
    polys = [shape(f["geometry"]) for f in feats]
    groups = contour_assign.cluster_polys_by_xgaps(polys, n_samples)
    if groups is None or len(groups) != n_samples:
        return None
    keep = set()
    for g in groups:
        amax = max(p.area for p in g)
        kept = [p for p in g if p.area >= debris_frac * amax] or g
        keep.update(id(p) for p in kept)
    return [i for i, p in enumerate(polys) if id(p) not in keep]


def main(dry_run=False):
    mapping = load_mapping()
    total_dropped = 0
    for stain, roots in ROOTS.items():
        id_col = "he" if stain == "hne" else "trichrome"
        for root in roots:
            cdir = root / "contours_geojson"
            for slide_id in FLAGGED[stain]:
                gj_path = cdir / f"{slide_id}.geojson"
                if not gj_path.exists():
                    print(f"  [{root.name}] {slide_id}: no geojson — skip")
                    continue
                row = mapping[mapping[id_col].astype("Int64") == slide_id].iloc[0]
                n_samples = len(vsr_ids_for(row))
                gj = json.loads(gj_path.read_text())
                drop_idx = debris_feature_indices(gj, n_samples, DEBRIS_FRAC)
                if drop_idx is None:
                    print(f"  [{root.name}] {slide_id}: cluster mismatch "
                          f"({len(gj['features'])} feats, {n_samples} samples) — SKIP")
                    continue
                if not drop_idx:
                    print(f"  [{root.name}] {slide_id}: no debris ({len(gj['features'])} feats kept)")
                    continue
                dropped_ids = [gj["features"][i].get("properties", {}).get("tissue_id", i)
                               for i in drop_idx]
                total_dropped += len(drop_idx)
                kept_feats = [f for i, f in enumerate(gj["features"]) if i not in set(drop_idx)]
                print(f"  [{root.name}] {slide_id}: dropping {len(drop_idx)} debris "
                      f"feature(s) tissue_id={dropped_ids} "
                      f"({len(gj['features'])} -> {len(kept_feats)})")
                if dry_run:
                    continue
                bdir = cdir / "_pre_debris_drop"
                bdir.mkdir(parents=True, exist_ok=True)
                if not (bdir / f"{slide_id}.geojson").exists():
                    shutil.copy2(gj_path, bdir / f"{slide_id}.geojson")
                gj["features"] = kept_feats
                gj_path.write_text(json.dumps(gj))
    print(f"\nDone. {'(dry) ' if dry_run else ''}dropped {total_dropped} debris features total."
          + ("" if dry_run else "  Backups in each contours_geojson/_pre_debris_drop/"))


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
