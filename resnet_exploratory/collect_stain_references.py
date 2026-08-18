#!/usr/bin/env python3
"""
Collect candidate stain normalization reference images.

Builds a library of reference thumbnails in output/stain_references/ for
comparing normalization quality across different reference choices.

Sources:
  1. From our data: thumbnails from SSc dataset slides (both HNE and trichrome)
  2. Public: standard H&E reference patches from publicly available sources

Usage:
    python collect_stain_references.py --stain hne
    python collect_stain_references.py --stain trichrome
    python collect_stain_references.py --stain hne --slides vsr261 vsr289 vsr300
    python collect_stain_references.py --stain hne --public   # download public refs
    python collect_stain_references.py --list                 # list existing refs
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import openslide
from PIL import Image

_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_REF_DIR = Path("/scratch/st-singha53-1/bhelfert/CBL/output/stain_references")

_WSI_DIRS = {
    "hne": _DATA / "hne_10_19_2023",
    "trichrome": _DATA / "trichrome_5_31_2024",
}

# Default slides to extract thumbnails from per stain
# (choose slides with representative, high-quality staining)
_DEFAULT_SLIDES = {
    "hne": ["vsr261085", "vsr289", "vsr300"],
    "trichrome": ["vsr289", "vsr261085", "vsr300"],
}

# Publicly available H&E reference images commonly used in normalization papers.
# These are small patches, so download is fast.
_PUBLIC_REFS = {
    # Macenko et al. 2009 reference image (H&E, lung tissue)
    # Mirrored via tiatoolbox's test assets
    "macenko_ref": "https://tiatoolbox.dcs.warwick.ac.uk/testdata/common/CMU-1-Small-Region.svs",
    # Reinhard et al. reference (TCGA-like, often used in tutorials)
    # Note: these are large WSIs; we download and extract a patch thumbnail
}

_THUMB_MAX_DIM = 2048  # max dimension for saved reference thumbnails


def extract_thumbnail(wsi_path: Path, max_dim: int = _THUMB_MAX_DIM,
                      min_tissue_fraction: float = 0.1) -> np.ndarray | None:
    """Extract a tissue-rich thumbnail from a WSI at ~max_dim resolution."""
    try:
        slide = openslide.OpenSlide(str(wsi_path))
    except Exception as e:
        print(f"  Could not open {wsi_path.name}: {e}")
        return None

    # Use the highest available pyramid level that still gives >= max_dim
    # on the longer side (or just level 0 if the slide is small)
    w0, h0 = slide.level_dimensions[0]
    scale = max_dim / max(w0, h0)

    if scale >= 1.0:
        level = 0
        w, h = w0, h0
    else:
        # Find the level closest to max_dim
        level = 0
        for lv in range(slide.level_count):
            wl, hl = slide.level_dimensions[lv]
            if max(wl, hl) <= max_dim:
                level = max(0, lv - 1)  # one level sharper
                break
        w, h = slide.level_dimensions[level]

    img = np.array(slide.read_region((0, 0), level, (w, h)).convert("RGB"))
    slide.close()

    # Check tissue content
    gray = np.mean(img, axis=2)
    tissue_fraction = np.mean(gray < 200)
    if tissue_fraction < min_tissue_fraction:
        print(f"  Warning: {wsi_path.name} has only {100*tissue_fraction:.1f}% tissue "
              f"at level {level} — may be a poor reference")

    return img


def find_slide(wsi_dir: Path, slide_id: str) -> Path | None:
    """Find a slide by partial name match."""
    for ext in (".svs", ".tif", ".tiff", ".ndpi"):
        matches = list(wsi_dir.glob(f"*{slide_id}*{ext}"))
        if matches:
            return matches[0]
    return None


def collect_from_our_data(stain: str, slide_ids: list[str]) -> list[Path]:
    """Extract and save thumbnails from our SSc dataset slides."""
    wsi_dir = _WSI_DIRS[stain]
    if not wsi_dir.exists():
        print(f"WSI directory not found: {wsi_dir}")
        return []

    saved = []
    for slide_id in slide_ids:
        slide_path = find_slide(wsi_dir, slide_id)
        if slide_path is None:
            print(f"  Slide not found: {slide_id} in {wsi_dir}")
            continue

        out_path = _REF_DIR / f"{slide_path.stem}_{stain}_thumbnail.png"
        if out_path.exists():
            print(f"  Already exists: {out_path.name}")
            saved.append(out_path)
            continue

        print(f"  Extracting: {slide_path.name} → {out_path.name}")
        img = extract_thumbnail(slide_path)
        if img is None:
            continue

        Image.fromarray(img).save(out_path)
        tissue_pct = np.mean(np.mean(img, axis=2) < 200) * 100
        print(f"    Saved: {out_path.name} ({img.shape[1]}x{img.shape[0]}, "
              f"{tissue_pct:.1f}% tissue)")
        saved.append(out_path)

    return saved


def collect_public_refs() -> list[Path]:
    """Download public H&E reference images."""
    saved = []
    for name, url in _PUBLIC_REFS.items():
        out_path = _REF_DIR / f"{name}.png"
        if out_path.exists():
            print(f"  Already exists: {out_path.name}")
            saved.append(out_path)
            continue

        # For WSI URLs, download and extract a thumbnail patch
        tmp_path = _REF_DIR / f"_tmp_{name}{Path(url).suffix}"
        print(f"  Downloading: {name} from {url}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
        except Exception as e:
            print(f"  Download failed: {e}")
            continue

        try:
            if tmp_path.suffix in (".svs", ".tif", ".tiff", ".ndpi"):
                img = extract_thumbnail(tmp_path)
            else:
                img = np.array(Image.open(tmp_path).convert("RGB"))

            if img is not None:
                Image.fromarray(img).save(out_path)
                tissue_pct = np.mean(np.mean(img, axis=2) < 200) * 100
                print(f"    Saved: {out_path.name} ({img.shape[1]}x{img.shape[0]}, "
                      f"{tissue_pct:.1f}% tissue)")
                saved.append(out_path)
        except Exception as e:
            print(f"  Processing failed: {e}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return saved


def list_refs():
    """Print all existing reference images with basic stats."""
    refs = sorted(_REF_DIR.glob("*.png"))
    if not refs:
        print(f"No reference images found in {_REF_DIR}")
        return

    print(f"Reference images in {_REF_DIR}:")
    print(f"  {'Name':<45} {'Size':>10}  {'Tissue%':>8}")
    print(f"  {'-'*45} {'-'*10}  {'-'*8}")
    for p in refs:
        try:
            img = np.array(Image.open(p).convert("RGB"))
            h, w = img.shape[:2]
            tissue_pct = np.mean(np.mean(img, axis=2) < 200) * 100
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:<45} {size_mb:>8.1f}MB  {tissue_pct:>7.1f}%")
        except Exception as e:
            print(f"  {p.name:<45} (error: {e})")


def main():
    parser = argparse.ArgumentParser(
        description="Collect stain normalization reference images"
    )
    parser.add_argument("--stain", choices=["hne", "trichrome"],
                        help="Stain type (required unless --list)")
    parser.add_argument("--slides", nargs="+",
                        help="Slide IDs to use (default: per-stain defaults)")
    parser.add_argument("--public", action="store_true",
                        help="Also download public reference images")
    parser.add_argument("--list", action="store_true",
                        help="List existing reference images and exit")
    args = parser.parse_args()

    _REF_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_refs()
        return

    if not args.stain:
        parser.error("--stain is required unless using --list")

    slides = args.slides or _DEFAULT_SLIDES.get(args.stain, [])

    print(f"\nCollecting {args.stain.upper()} reference images...")
    print(f"Output directory: {_REF_DIR}\n")

    our_refs = collect_from_our_data(args.stain, slides)
    print(f"\nFrom our data: {len(our_refs)} references saved")

    if args.public:
        print("\nDownloading public references...")
        pub_refs = collect_public_refs()
        print(f"Public references: {len(pub_refs)} saved")

    print("\nDone. Current references:")
    list_refs()


if __name__ == "__main__":
    main()
