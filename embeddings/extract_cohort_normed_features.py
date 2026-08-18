#!/usr/bin/env python3
"""Stage 2 of the new normalized-embedding pipeline.

Re-encode TRIDENT tissue patches after normalizing each patch from its slide's
source stain stats to the cohort-median target (computed by
compute_cohort_stain_reference.py):

  H&E        Macenko  (cohort-median [H;E] vectors + max-conc)
  Trichrome  Reinhard (cohort-median LAB mean/std)

Only TRIDENT tissue patches are touched (reuses existing coords), so the mostly-
blank slide area never enters the embeddings -- same "tissue patches only"
behaviour as the previous normed-embedding run, just a different normalizer and a
cohort-median reference instead of one hand-picked slide.

Output h5 (TRIDENT format: coords + features) goes to a NEW sibling dir so the
old features_* are never clobbered:
  H&E        features_<enc>_macenko/<slide>.h5
  Trichrome  features_<enc>_reinhardmed/<slide>.h5

GPU.  Run via run_cohort_normed_features.sh.

  python extract_cohort_normed_features.py --encoder resnet50 --stain hne
  python extract_cohort_normed_features.py --encoder uni_v1   --stain hne
  python extract_cohort_normed_features.py --encoder resnet50 --stain trichrome
  python extract_cohort_normed_features.py --encoder conch_v1 --stain trichrome
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_PREPROC = Path(__file__).resolve().parent
sys.path.insert(0, str(_PREPROC))
import cohort_norm as cn  # noqa: E402

_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_CBL = Path("/scratch/st-singha53-1/bhelfert/CBL")
FEAT = _CBL / "output" / "features"
REF_DIR = _CBL / "output" / "stain_references"

STAIN_WSI = {"hne": _DATA / "hne_10_19_2023",
             "trichrome": _DATA / "trichrome_5_31_2024"}

# encoder + stain -> TRIDENT root/patch grid that already has coords
ENC = {
    ("resnet50", "hne"):     dict(root=FEAT / "resnet50_hne_20x",     sub="20x_256px_0px_overlap"),
    ("uni_v1", "hne"):       dict(root=FEAT / "uni_v1_hne",           sub="20x_256px_0px_overlap"),
    ("resnet50", "trichrome"): dict(root=FEAT / "resnet50_trichrome_20x", sub="20x_256px_0px_overlap"),
    ("conch_v1", "trichrome"): dict(root=FEAT / "conch_v1_trichrome",     sub="20x_512px_0px_overlap"),
}
NORM_TAG = {"hne": "vahadane_med", "trichrome": "reinhardmed"}


class PatchNormDataset(Dataset):
    """Reads patches via WSIPatcher, normalizes each (closure), applies transform."""

    def __init__(self, patcher, norm_fn, transform):
        self.patcher, self.norm_fn, self.transform = patcher, norm_fn, transform

    def __len__(self):
        return len(self.patcher)

    def __getitem__(self, i):
        tile, x, y = self.patcher[i]
        arr = np.asarray(tile)[:, :, :3]
        arr = self.norm_fn(arr)
        pil = Image.fromarray(arr)
        if self.transform:
            pil = self.transform(pil)
        return pil, (x, y)


def load_reference(stain):
    ref = np.load(REF_DIR / f"cohort_stain_reference_{stain}.npz", allow_pickle=True)
    ids = [str(s) for s in ref["slide_ids"]]
    if stain == "hne":
        src = {sid: (ref["src_sm"][i], ref["src_mc"][i]) for i, sid in enumerate(ids)}
        tgt = (ref["target_sm"], ref["target_mc"])
    else:
        src = {sid: (ref["src_mean"][i], ref["src_std"][i]) for i, sid in enumerate(ids)}
        tgt = (ref["target_mean"], ref["target_std"])
    return src, tgt


def make_norm_fn(stain, slide_src, tgt):
    if stain == "hne":
        src_sm, src_mc = slide_src
        tgt_sm, tgt_mc = tgt
        return lambda a: cn.vahadane_transform(a, src_sm, src_mc, tgt_sm, tgt_mc)
    src_mean, src_std = slide_src
    tgt_mean, tgt_std = tgt
    return lambda a: cn.reinhard_transform(a, src_mean, src_std, tgt_mean, tgt_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["resnet50", "uni_v1", "conch_v1"])
    ap.add_argument("--stain", required=True, choices=["hne", "trichrome"])
    ap.add_argument("--slide_name", default=None)
    ap.add_argument("--array_index", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    key = (args.encoder, args.stain)
    if key not in ENC:
        sys.exit(f"unsupported encoder/stain combo {key}")
    cfg = ENC[key]
    coords_dir = cfg["root"] / cfg["sub"]
    patches_dir = coords_dir / "patches"
    out_dir = coords_dir / f"features_{args.encoder}_{NORM_TAG[args.stain]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    wsi_dir = STAIN_WSI[args.stain]

    src_map, tgt = load_reference(args.stain)
    print(f"encoder={args.encoder} stain={args.stain}")
    print(f"coords:  {coords_dir}")
    print(f"output:  {out_dir}")

    all_slides = sorted(p.stem for p in wsi_dir.glob("*.svs"))
    if args.slide_name:
        slides = [args.slide_name]
    elif args.array_index is not None:
        slides = [all_slides[args.array_index]]
    else:
        slides = all_slides
    print(f"slides:  {len(slides)}")

    from trident import load_wsi, WSIPatcher
    from trident.IO import read_coords, save_h5
    from trident.patch_encoder_models.load import encoder_factory

    print(f"loading encoder {args.encoder} ...")
    encoder = encoder_factory(args.encoder)
    encoder.to(args.device).eval()
    precision = getattr(encoder, "precision", torch.float32)
    tfm = encoder.eval_transforms
    print(f"  precision={precision}")

    for idx, sid in enumerate(slides):
        out_h5 = out_dir / f"{sid}.h5"
        if out_h5.exists() and not args.overwrite:
            print(f"[{idx+1}/{len(slides)}] {sid}: exists — skip")
            continue
        coords_path = patches_dir / f"{sid}_patches.h5"
        if not coords_path.exists():
            print(f"[{idx+1}/{len(slides)}] {sid}: no coords — skip")
            continue
        if sid not in src_map:
            print(f"[{idx+1}/{len(slides)}] {sid}: no source stats in reference — skip")
            continue

        print(f"\n[{idx+1}/{len(slides)}] {sid}")
        t0 = time.time()
        wsi = load_wsi(slide_path=str(wsi_dir / f"{sid}.svs"), name=sid)
        wsi._lazy_initialize()
        coords_attrs, coords = read_coords(str(coords_path))
        level0_mag = coords_attrs.get("level0_magnification", 40)
        target_mag = coords_attrs.get("target_magnification", 20)
        patch_size = int(coords_attrs.get("patch_size", 256))
        print(f"  {len(coords)} patches  {target_mag}x {patch_size}px")

        patcher = WSIPatcher(wsi, patch_size=patch_size, src_mag=level0_mag,
                             dst_mag=target_mag, custom_coords=coords,
                             coords_only=False, pil=False)
        norm_fn = make_norm_fn(args.stain, src_map[sid], tgt)
        loader = DataLoader(PatchNormDataset(patcher, norm_fn, tfm),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True)

        feats = []
        for imgs, _ in tqdm(loader, desc="  encoding", leave=False):
            imgs = imgs.to(args.device)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=precision,
                                    enabled=(precision != torch.float32)):
                    feats.append(encoder(imgs).cpu().numpy())
        feats = np.concatenate(feats, axis=0)

        save_h5(str(out_h5),
                assets={"features": feats, "coords": coords},
                attributes={"features": {"name": sid, "encoder": args.encoder,
                                         "normalization": NORM_TAG[args.stain],
                                         "reference": "cohort_median"},
                            "coords": coords_attrs},
                mode="w")
        wsi.release()
        print(f"  saved {feats.shape} -> {out_h5.name}  ({time.time()-t0:.0f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
