"""Assign embedding patches to samples by TRIDENT-contour overlap.

Treats the TRIDENT polygons as ground truth instead of the 1-D patch x-gap split
(which mixes samples when tissue pieces overlap in x, e.g. trichrome 269311
vsr344/vsr346). Steps:

  1. Cluster polygons into n_samples groups by centroid x-gap, label L->R
     (handles a sample fragmented into several polygons).
  2. Drop debris polygons: area < debris_frac * (largest polygon in that cluster).
  3. Assign each patch to the sample whose tissue it OVERLAPS most; a patch that
     overlaps no real tissue is dropped.

Because assignment is by overlap with the kept (real) contours, debris is excluded
automatically: drop the debris polygons from the contour file and the patches that
sat on them simply overlap nothing and are dropped. No special debris bookkeeping,
no need to keep the original contours around.

Used by repool_contour_embeddings.py, add_vsr346.py, and qc_sample_assignment.py
(--corrected overlay).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import box, shape
from shapely.ops import unary_union


def load_polys_level0(geojson_path: Path) -> list:
    """Flat list of shapely polygons in level-0 px (MultiPolygons expanded)."""
    gj = json.loads(Path(geojson_path).read_text())
    flat = []
    for f in gj["features"]:
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        if g.geom_type == "MultiPolygon":
            flat += [p for p in g.geoms if not p.is_empty]
        else:
            flat.append(g)
    return flat


def cluster_polys_by_xgaps(polys: list, n: int):
    """Cluster polygons into n groups by centroid-x gaps, sorted L->R. None if too few."""
    if n <= 1:
        return [list(polys)]
    if len(polys) < n:
        return None
    cx = np.array([p.centroid.x for p in polys])
    order = np.argsort(cx)
    sx = cx[order]
    cuts = np.sort(np.argsort(np.diff(sx))[::-1][: n - 1])
    groups, prev = [], 0
    for c in cuts:
        groups.append([polys[order[i]] for i in range(prev, c + 1)])
        prev = c + 1
    groups.append([polys[order[i]] for i in range(prev, len(polys))])
    groups.sort(key=lambda g: float(np.mean([p.centroid.x for p in g])))
    return groups


def assign_patches(coords_l0: np.ndarray, patch_l0: float, geojson_path: Path,
                   vsr_ids: list[str], debris_frac: float = 0.05):
    """Overlap-based patch->sample assignment; off-tissue patches dropped.

    Returns (assign, kept_polys, info):
      assign     : dict vsr_id -> np.ndarray of patch indices
      kept_polys : dict vsr_id -> list[Polygon] (debris dropped, for display)
      info       : dict with status / counts (status='ok' on success)
    """
    polys = load_polys_level0(geojson_path)
    groups = cluster_polys_by_xgaps(polys, len(vsr_ids))
    if groups is None or len(groups) != len(vsr_ids):
        return None, None, {"status": "cluster_mismatch",
                            "n_polys": len(polys), "n_samples": len(vsr_ids)}

    kept_polys, dropped = {}, {}
    for vsr, g in zip(vsr_ids, groups):
        amax = max(p.area for p in g)
        keep = [p for p in g if p.area >= debris_frac * amax] or g  # never empty
        kept_polys[vsr] = keep
        dropped[vsr] = len(g) - len(keep)

    sample_union = {v: unary_union(ps) for v, ps in kept_polys.items()}

    assign = {v: [] for v in vsr_ids}
    n_offtissue = 0
    for i, (x, y) in enumerate(coords_l0):
        b = box(float(x), float(y), float(x) + patch_l0, float(y) + patch_l0)
        best_v, best_ov = None, 0.0
        for v, u in sample_union.items():
            ov = u.intersection(b).area
            if ov > best_ov:
                best_ov, best_v = ov, v
        if best_v is None:                     # patch overlaps no real tissue
            n_offtissue += 1
        else:
            assign[best_v].append(i)
    assign = {v: np.asarray(ix, dtype=int) for v, ix in assign.items()}

    info = {"status": "ok", "n_polys": len(polys),
            "dropped_debris": dropped, "n_patches": int(len(coords_l0)),
            "n_offtissue_dropped": int(n_offtissue),
            "n_per_sample": {v: int(len(ix)) for v, ix in assign.items()}}
    return assign, kept_polys, info
