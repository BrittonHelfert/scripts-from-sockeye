"""
Extract patch features with stain normalization.

Uses TRIDENT's existing tissue segmentation and patch coordinates.
For each slide: reads patches from WSI, normalizes stain, encodes with
a foundation model, and saves features in h5 format (same format as
TRIDENT's native output).

Normalization method:
    H&E (hne):        Vahadane (sparse NMF, 2-stain decomposition)
    Trichrome:        Reinhard (LAB color transfer, stain-agnostic)

Usage:
    # Single slide
    python extract_normalized_features.py --encoder uni_v1 --stain hne --slide_name 261085

    # All slides (default)
    python extract_normalized_features.py --encoder uni_v1 --stain hne

    # SLURM array (one slide per task)
    python extract_normalized_features.py --encoder uni_v1 --stain hne --array_index $SLURM_ARRAY_TASK_ID

    # Custom reference image
    python extract_normalized_features.py --encoder conch_v1 --stain trichrome --stain_ref /path/to/ref.png
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add scripts dir for stain_normalize
sys.path.insert(0, str(Path(__file__).parent))
from stain_normalize import VahadaneNormalizer, ReinhardNormalizer, _get_tissue_mask

# --- Paths ---
_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_FEAT = Path("/scratch/st-singha53-1/bhelfert/CBL/output/features")
_REF_DIR = Path("/scratch/st-singha53-1/bhelfert/CBL/output/stain_references")

STAIN_CONFIG = {
    "hne": {
        "wsi_dir": _DATA / "hne_10_19_2023",
        "reference": _REF_DIR / "vsr289_thumbnail.png",
    },
    "trichrome": {
        "wsi_dir": _DATA / "trichrome_5_31_2024",
        "reference": _REF_DIR / "vsr289_trichrome_thumbnail.png",
    },
}

# Map encoder names to the TRIDENT job dirs that have existing coords
ENCODER_CONFIG = {
    "uni_v1": {
        "hne": {"job_dir": _FEAT / "uni_v1_hne", "patch_size": 256, "mag": 20},
        "trichrome": None,
    },
    "conch_v1": {
        "hne": {"job_dir": _FEAT / "conch_v1_hne", "patch_size": 512, "mag": 20},
        "trichrome": {"job_dir": _FEAT / "conch_v1_trichrome", "patch_size": 512, "mag": 20},
    },
    "resnet50": {
        "hne": {"job_dir": _FEAT / "resnet50_hne_20x", "patch_size": 256, "mag": 20},
        "trichrome": {"job_dir": _FEAT / "resnet50_trichrome_20x", "patch_size": 256, "mag": 20},
    },
    "gigapath": {
        "hne": {"job_dir": _FEAT / "gigapath_hne", "patch_size": 256, "mag": 20},
        "trichrome": None,
    },
}


def save_qc_figures(slide_name, thumb, raw_patches, norm_patches, qc_dir):
    """Save tissue mask overlay and before/after normalization comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [QC] matplotlib not available, skipping QC figures")
        return

    qc_dir.mkdir(parents=True, exist_ok=True)
    tissue_mask = _get_tissue_mask(thumb)

    # --- Tissue mask overlay ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(thumb)
    axes[0].set_title("Thumbnail")
    axes[0].axis("off")

    overlay = thumb.copy().astype(float)
    overlay[tissue_mask, 0] = np.clip(overlay[tissue_mask, 0] * 0.4, 0, 255)
    overlay[tissue_mask, 1] = np.clip(overlay[tissue_mask, 1] * 0.4 + 120, 0, 255)
    overlay[tissue_mask, 2] = np.clip(overlay[tissue_mask, 2] * 0.4, 0, 255)
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"Tissue mask (green = {100*tissue_mask.mean():.1f}% of image)")
    axes[1].axis("off")

    fig.suptitle(f"{slide_name} — Tissue Segmentation", fontsize=13)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{slide_name}_tissue_mask.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # --- Before / after normalization grid ---
    n = min(len(raw_patches), 8)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        axes[0, i].imshow(raw_patches[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(norm_patches[i])
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Before", fontsize=11)
    axes[1, 0].set_ylabel("After", fontsize=11)

    fig.suptitle(f"{slide_name} — Stain Normalization Comparison", fontsize=13)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{slide_name}_norm_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved tissue mask + normalization comparison → {qc_dir.name}/")


class NormalizedPatchDataset(Dataset):
    """Dataset that reads patches from WSI, normalizes, and applies transforms."""

    def __init__(self, patcher, normalizer, transform):
        self.patcher = patcher
        self.normalizer = normalizer
        self.transform = transform

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, index):
        tile, x, y = self.patcher[index]
        tile_np = np.array(tile) if not isinstance(tile, np.ndarray) else tile
        tile_np = tile_np[:, :, :3]

        # Normalize
        tile_np = self.normalizer.transform(tile_np)

        # Convert to PIL for transforms (TRIDENT encoders expect PIL input)
        tile_pil = Image.fromarray(tile_np)
        if self.transform:
            tile_pil = self.transform(tile_pil)

        return tile_pil, (x, y)


def get_slide_thumbnail(wsi, max_dim=4096):
    """Get thumbnail from TRIDENT WSI object."""
    w, h = wsi.get_dimensions()
    scale = min(max_dim / max(w, h), 1.0)
    thumb_w, thumb_h = int(w * scale), int(h * scale)
    thumb = wsi.get_thumbnail((thumb_w, thumb_h))
    return np.array(thumb)[:, :, :3]


def find_coords_dir(job_dir, mag, patch_size):
    """Find the coords directory within a TRIDENT job dir."""
    # TRIDENT naming: {mag}x_{patch_size}px_{overlap}px_overlap
    for d in sorted(job_dir.iterdir()):
        if d.is_dir() and d.name.startswith(f"{mag}x_{patch_size}px"):
            return d
    raise FileNotFoundError(
        f"No coords dir matching {mag}x_{patch_size}px* in {job_dir}")


def get_slide_names(wsi_dir):
    """Get sorted list of slide names (without extension)."""
    return sorted(p.stem for p in wsi_dir.glob("*.svs"))


def main():
    parser = argparse.ArgumentParser(
        description="Extract features with stain normalization (Vahadane for H&E, Reinhard for trichrome)")
    parser.add_argument("--encoder", required=True,
                        choices=list(ENCODER_CONFIG.keys()),
                        help="Patch encoder model")
    parser.add_argument("--stain", required=True,
                        choices=list(STAIN_CONFIG.keys()),
                        help="Stain type")
    parser.add_argument("--stain_ref", default=None,
                        help="Override default reference image path (PNG)")
    parser.add_argument("--slide_name", default=None,
                        help="Process a single slide (name without .svs)")
    parser.add_argument("--array_index", type=int, default=None,
                        help="SLURM array index (processes one slide)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_qc", action="store_true",
                        help="Skip saving QC figures (tissue mask + normalization comparison)")
    args = parser.parse_args()

    # Validate config
    enc_cfg = ENCODER_CONFIG[args.encoder].get(args.stain)
    if enc_cfg is None:
        print(f"Error: no coords available for {args.encoder} + {args.stain}")
        sys.exit(1)

    stain_cfg = STAIN_CONFIG[args.stain]
    wsi_dir = stain_cfg["wsi_dir"]
    ref_path = Path(args.stain_ref) if args.stain_ref else stain_cfg["reference"]
    job_dir = enc_cfg["job_dir"]
    patch_size = enc_cfg["patch_size"]
    mag = enc_cfg["mag"]

    coords_dir = find_coords_dir(job_dir, mag, patch_size)
    patches_dir = coords_dir / "patches"

    # Output dir name reflects normalizer used
    norm_tag = "reinhard" if args.stain == "trichrome" else "vahadane"
    out_dir = coords_dir / f"features_{args.encoder}_{norm_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoder: {args.encoder}")
    print(f"Stain: {args.stain}")
    print(f"Coords: {coords_dir}")
    print(f"Output: {out_dir}")

    # Determine slides to process
    all_slides = get_slide_names(wsi_dir)
    if args.slide_name:
        slides = [args.slide_name]
    elif args.array_index is not None:
        slides = [all_slides[args.array_index]]
    else:
        slides = all_slides
    print(f"Slides to process: {len(slides)}")

    # Load encoder
    from trident.patch_encoder_models.load import encoder_factory
    print(f"Loading encoder {args.encoder}...")
    encoder = encoder_factory(args.encoder)
    encoder.to(args.device)
    encoder.eval()
    precision = getattr(encoder, "precision", torch.float32)
    patch_transforms = encoder.eval_transforms
    print(f"  Encoder loaded, precision={precision}")

    # Fit normalizer on reference (Vahadane for H&E, Reinhard for trichrome)
    print(f"Fitting normalizer on reference: {ref_path.name}")
    ref_img = np.array(Image.open(ref_path))[:, :, :3]
    if args.stain == "trichrome":
        normalizer = ReinhardNormalizer()
        print(f"  Normalizer: Reinhard (LAB color transfer)")
    else:
        normalizer = VahadaneNormalizer()
        print(f"  Normalizer: Vahadane (sparse NMF)")
    normalizer.fit(ref_img)

    # Import TRIDENT utilities
    from trident import load_wsi, WSIPatcher
    from trident.IO import read_coords, save_h5

    # Process slides
    for slide_idx, slide_name in enumerate(slides):
        slide_path = wsi_dir / f"{slide_name}.svs"
        coords_path = patches_dir / f"{slide_name}_patches.h5"
        out_path = out_dir / f"{slide_name}.h5"

        if out_path.exists():
            print(f"[{slide_idx+1}/{len(slides)}] {slide_name}: already exists, skipping")
            continue

        if not coords_path.exists():
            print(f"[{slide_idx+1}/{len(slides)}] {slide_name}: no coords, skipping")
            continue

        print(f"\n[{slide_idx+1}/{len(slides)}] {slide_name}")
        t0 = time.time()

        # Load WSI
        wsi = load_wsi(slide_path=str(slide_path), name=slide_name)
        wsi._lazy_initialize()

        # Fit source from thumbnail
        thumb = get_slide_thumbnail(wsi, max_dim=4096)
        normalizer.fit_source(thumb)
        if not normalizer.source_fitted:
            print(f"  Warning: could not fit source stats, skipping")
            wsi.release()
            continue

        # Load coords
        coords_attrs, coords = read_coords(str(coords_path))
        level0_mag = coords_attrs.get("level0_magnification", 40)
        target_mag = coords_attrs.get("target_magnification", mag)
        print(f"  {len(coords)} patches, {target_mag}x, {patch_size}px")

        # Create patcher
        patcher = WSIPatcher(
            wsi,
            patch_size=patch_size,
            src_mag=level0_mag,
            dst_mag=target_mag,
            custom_coords=coords,
            coords_only=False,
            pil=False,
        )

        # QC: sample patches before and after normalization
        if not args.no_qc:
            n_qc = min(8, len(patcher))
            step = max(1, len(patcher) // n_qc)
            qc_indices = list(range(0, len(patcher), step))[:n_qc]
            raw_patches, norm_patches = [], []
            for qi in qc_indices:
                tile, _, _ = patcher[qi]
                tile_np = np.array(tile)[:, :, :3] if not isinstance(tile, np.ndarray) else tile[:, :, :3]
                raw_patches.append(tile_np.copy())
                norm_patches.append(normalizer.transform(tile_np.copy()))
            qc_dir = out_dir.parent / "qc"
            save_qc_figures(slide_name, thumb, raw_patches, norm_patches, qc_dir)

        # Create dataset and dataloader
        dataset = NormalizedPatchDataset(patcher, normalizer, patch_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Extract features
        features = []
        for imgs, _ in tqdm(dataloader, desc=f"  Encoding", leave=False):
            imgs = imgs.to(args.device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type="cuda",
                    dtype=precision,
                    enabled=(precision != torch.float32),
                ):
                    batch_feats = encoder(imgs)
            features.append(batch_feats.cpu().numpy())

        features = np.concatenate(features, axis=0)

        # Save in same format as TRIDENT
        enc_name = getattr(encoder, "enc_name", args.encoder)
        save_h5(
            str(out_path),
            assets={"features": features, "coords": coords},
            attributes={
                "features": {
                    "name": slide_name,
                    "savetodir": str(out_dir),
                    "encoder": enc_name,
                    "normalization": norm_tag,
                },
                "coords": coords_attrs,
            },
            mode="w",
        )

        elapsed = time.time() - t0
        print(f"  Saved {features.shape} to {out_path.name} ({elapsed:.1f}s)")

        wsi.release()

    print("\nDone.")


if __name__ == "__main__":
    main()
