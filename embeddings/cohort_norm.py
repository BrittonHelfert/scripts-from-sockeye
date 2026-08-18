"""Cohort-median stain normalization for patch-level embedding extraction.

  H&E        -> Vahadane (sparse-NMF 2-stain decomposition)   [per-slide stain matrix]
  Trichrome  -> Reinhard (LAB channel-wise transfer)          [per-slide LAB mean/std]

The normalization TARGET is the element-wise cohort MEDIAN of the per-slide stain
statistics, estimated from TRIDENT tissue patches only (background masked out), so
the mostly-blank slide area never skews the estimate.

  Stage 1  compute_cohort_stain_reference.py  -> per-slide source stats + cohort median
  Stage 2  extract_cohort_normed_features.py  -> normalize each patch source->median,
                                                 then encode.

NOTE ON MACENKO (originally requested for H&E, per biomni_scripts):
  Macenko is DEGENERATE on this cohort. Skin dermis is collagen/eosin-dominated
  with sparse, weak hematoxylin, so the Macenko angle method returns near-collinear
  H/E vectors (angle ~0.5-2 deg, cohort-wide at every resolution). Its 2-stain
  projection then collapses patches to gray (no clip) or over-darkens them (clip),
  which would corrupt the embeddings. Vahadane's sparse NMF recovers well-separated
  H/E (angle ~17 deg) and is the method behind the project's prior headline result,
  so H&E uses Vahadane to the cohort median. See
  output/stain_references/NORMALIZATION_NOTES.md for the diagnostic evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage import color as skcolor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Reuse the tested Vahadane primitives (sparse-NMF stain matrix, OD helpers).
from stain_normalize import (  # noqa: E402
    _get_stain_matrix, _get_concentrations, _od_to_rgb,
)


def tissue_mask_gray(rgb: np.ndarray, threshold: int = 220) -> np.ndarray:
    """Intra-patch tissue mask: pixels darker than `threshold` in grayscale (0-255)."""
    gray = rgb[..., :3].mean(axis=-1)
    return gray < threshold


# ── Vahadane (H&E) ──────────────────────────────────────────────────────────────

def vahadane_stats(pixels: np.ndarray):
    """Per-slide Vahadane stain matrix (2,3 = [H;E]) + 99th-pct max-conc (2,).

    `pixels` is a flat (N,3) uint8 array of tissue pixels.
    """
    pix = pixels.reshape(-1, 3)
    stain_matrix = _get_stain_matrix(pix)              # (2,3), rows unit-norm
    conc = _get_concentrations(pix, stain_matrix)      # (N,2), >=0
    max_conc = np.percentile(conc, 99, axis=0)
    max_conc[max_conc == 0] = 1.0
    return stain_matrix, max_conc


def vahadane_transform(rgb, src_sm, src_mc, tgt_sm, tgt_mc, bg_threshold: int = 220):
    """Normalize an RGB patch source->target via Vahadane concentrations.

    Decompose with the slide's own stain matrix, rescale concentrations from the
    slide's max-conc to the cohort-median max-conc, reconstruct with the cohort-
    median stain matrix. Background pixels (grayscale >= bg_threshold) pass through.
    """
    rgb = rgb[..., :3]
    mask = tissue_mask_gray(rgb, bg_threshold)
    if not mask.any():
        return rgb.copy()
    out = rgb.copy()
    conc = _get_concentrations(rgb[mask], src_sm)      # (Nt,2)
    conc = conc * (tgt_mc / src_mc)
    od = conc @ tgt_sm                                  # (Nt,3)
    out[mask] = _od_to_rgb(od)
    return out


# ── Reinhard (trichrome / any stain) ───────────────────────────────────────────

def reinhard_stats(pixels: np.ndarray):
    """LAB mean/std (each (3,)) from a flat (N,3) uint8 tissue-pixel array."""
    lab = skcolor.rgb2lab(pixels.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def reinhard_transform(rgb, src_mean, src_std, tgt_mean, tgt_std, bg_threshold: int = 220):
    """Reinhard LAB transfer on tissue pixels; background passed through."""
    rgb = rgb[..., :3]
    mask = tissue_mask_gray(rgb, bg_threshold)
    if not mask.any():
        return rgb.copy()
    lab = skcolor.rgb2lab(rgb.astype(np.float64) / 255.0)
    for ch in range(3):
        lab[mask, ch] = ((lab[mask, ch] - src_mean[ch]) / src_std[ch]
                         * tgt_std[ch] + tgt_mean[ch])
    out = (np.clip(skcolor.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
    out[~mask] = rgb[~mask]
    return out


# ── Tissue-pixel sampling from TRIDENT patches (full resolution) ────────────────

def patch_tissue_pixels(coords_h5: Path, svs_path: Path, n_patches: int = 150,
                        max_pixels: int = 400_000, bg_threshold: int = 220,
                        seed: int = 0):
    """Pool tissue RGB pixels from a sample of the slide's TRIDENT patches.

    Reads each sampled patch at level 0 (patch_size_level0 px) so stain structure
    is at full resolution -- essential for Vahadane/Macenko stain separation, which
    is washed out by the heavily downsampled whole-slide thumbnail.

    Returns (pixels (N,3) uint8, n_patches_used).
    """
    import h5py
    import openslide

    with h5py.File(coords_h5, "r") as f:
        coords = f["coords"][:]
        patch_l0 = int(f["coords"].attrs["patch_size_level0"])

    rng = np.random.default_rng(seed)
    n = min(n_patches, len(coords))
    idx = rng.choice(len(coords), size=n, replace=False)

    sl = openslide.OpenSlide(str(svs_path))
    pooled = []
    used = 0
    for i in idx:
        x, y = int(coords[i][0]), int(coords[i][1])
        tile = np.array(sl.read_region((x, y), 0, (patch_l0, patch_l0)).convert("RGB"))
        m = tissue_mask_gray(tile, bg_threshold)
        if m.sum() == 0:
            continue
        pooled.append(tile[m])
        used += 1
    sl.close()
    if not pooled:
        return np.empty((0, 3), np.uint8), 0
    px = np.concatenate(pooled, axis=0)
    if len(px) > max_pixels:
        px = px[rng.choice(len(px), max_pixels, replace=False)]
    return px, used
