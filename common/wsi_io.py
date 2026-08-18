"""WSI loading, GeoJSON parsing, polygon-to-sample assignment, mask rasterization."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import openslide
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import shape as shapely_shape
from shapely.affinity import scale as shapely_scale
from skimage.draw import polygon as draw_polygon


def open_slide_at_level(svs_path: Path, level: int = 1):
    """Load WSI at the requested pyramid level, returning RGB array, downsample, mpp."""
    slide = openslide.OpenSlide(str(svs_path))
    level = min(level, slide.level_count - 1)
    w, h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    mpp_x_str = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp_x_str is None:
        raise ValueError(f"{svs_path.name} has no MPP property")
    mpp_level0 = float(mpp_x_str)
    mpp_level = mpp_level0 * downsample

    img = np.array(slide.read_region((0, 0), level, (w, h)).convert("RGB"))
    slide.close()
    return img, downsample, mpp_level


def load_tissue_polygons(geojson_path: Path, downsample: float) -> list:
    """Parse TRIDENT GeoJSON; return shapely polygons in level-1 pixel coordinates.

    GeoJSON coordinates are in level-0 pixel space; we divide by the level-0→level-1
    downsample factor. Returns a flat list of Polygon objects (MultiPolygons are
    expanded into their constituent polygons).
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    inv = 1.0 / downsample
    polys = []
    for feat in gj["features"]:
        geom = shapely_shape(feat["geometry"])
        scaled = shapely_scale(geom, xfact=inv, yfact=inv, origin=(0, 0, 0))
        if isinstance(scaled, Polygon):
            if not scaled.is_empty:
                polys.append(scaled)
        elif isinstance(scaled, MultiPolygon):
            for p in scaled.geoms:
                if not p.is_empty:
                    polys.append(p)
    return polys


def cluster_polygons_by_x_gaps(polys: list, n_pieces: int) -> list[list]:
    """Cluster polygons into n_pieces groups using gaps in their centroid x-coordinates.

    Mirrors export_vahadane_matrices._cluster_patches_by_x_gaps but operates on
    polygon centroids: sort polygons by centroid x, find the (n_pieces - 1) largest
    gaps between consecutive centroids, split there, return groups sorted by mean
    centroid x (left-to-right). Returns one list of polygons per cluster.
    """
    if n_pieces <= 0:
        return []
    if len(polys) == 0:
        return [[] for _ in range(n_pieces)]
    if n_pieces == 1:
        return [list(polys)]

    centroids_x = np.array([p.centroid.x for p in polys])
    sort_idx = np.argsort(centroids_x)
    sorted_polys = [polys[i] for i in sort_idx]
    sorted_x = centroids_x[sort_idx]

    if len(polys) < n_pieces:
        return None  # cannot split — caller decides what to do

    gaps = np.diff(sorted_x)
    gap_positions = np.sort(np.argsort(gaps)[::-1][:n_pieces - 1])

    groups = []
    prev = 0
    for gp in gap_positions:
        groups.append(sorted_polys[prev:gp + 1])
        prev = gp + 1
    groups.append(sorted_polys[prev:])
    groups.sort(key=lambda g: np.mean([p.centroid.x for p in g]))
    return groups


def assign_polygons_to_samples(polys: list, vsr_ids: list[str],
                                logger: logging.Logger | None = None) -> dict | None:
    """Cluster polygons into len(vsr_ids) groups, assign left-to-right to vsr_ids.

    Returns dict[vsr_id -> list[Polygon]] or None on cluster-count mismatch.
    """
    n_pieces = len(vsr_ids)
    groups = cluster_polygons_by_x_gaps(polys, n_pieces)
    if groups is None or len(groups) != n_pieces:
        if logger:
            n_found = 0 if groups is None else len(groups)
            logger.error(
                f"MISMATCH: clustered {len(polys)} polygons into {n_found} groups, "
                f"expected {n_pieces} for vsr_ids={vsr_ids}"
            )
        return None
    return dict(zip(vsr_ids, groups))


def rasterize_polygons(polys: list, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a list of shapely polygons into a bool mask of given (H, W) shape.

    Handles polygons with holes (interior rings are removed from the mask).
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for poly in polys:
        if poly.is_empty:
            continue
        # Outer ring
        ext_x, ext_y = poly.exterior.coords.xy
        rr, cc = draw_polygon(np.array(ext_y), np.array(ext_x), shape=(h, w))
        mask[rr, cc] = True
        # Inner rings (holes)
        for interior in poly.interiors:
            int_x, int_y = interior.coords.xy
            rr, cc = draw_polygon(np.array(int_y), np.array(int_x), shape=(h, w))
            mask[rr, cc] = False
    return mask


def polygons_bounding_box(polys: list, shape: tuple[int, int],
                           pad: int = 16) -> tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) bbox of polygons, clipped to image shape."""
    h, w = shape
    xmins, ymins, xmaxs, ymaxs = [], [], [], []
    for p in polys:
        if p.is_empty:
            continue
        x0, y0, x1, y1 = p.bounds
        xmins.append(x0); ymins.append(y0); xmaxs.append(x1); ymaxs.append(y1)
    if not xmins:
        return 0, h, 0, w
    rmin = max(0, int(np.floor(min(ymins))) - pad)
    rmax = min(h, int(np.ceil(max(ymaxs))) + pad)
    cmin = max(0, int(np.floor(min(xmins))) - pad)
    cmax = min(w, int(np.ceil(max(xmaxs))) + pad)
    return rmin, rmax, cmin, cmax
