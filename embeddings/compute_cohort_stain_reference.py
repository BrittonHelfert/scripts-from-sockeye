#!/usr/bin/env python3
"""Stage 1 of the new normalized-embedding pipeline.

For one stain, estimate each slide's source stain statistics from its TRIDENT
tissue patches (sampled at full level-0 resolution, background masked), then take
the element-wise cohort MEDIAN as the normalization target:

  H&E        Vahadane -> per-slide stain matrix (2,3 = [H;E]) + max-conc (2,)
                         target = median over slides (rows re-unit-normed)
  Trichrome  Reinhard -> per-slide LAB mean (3,) + std (3,)
                         target = median over slides

Both per-slide source stats AND the cohort-median target are saved to one .npz so
Stage 2 normalizes patches source->median without re-reading anything.

Stain stats are estimated from PATCHES (not the whole-slide thumbnail): the
thumbnail is ~30-50x downsampled, which washes out nuclear/eosin structure and
makes stain separation degenerate. (This is why Macenko -- requested originally --
fails here; see NORMALIZATION_NOTES.md.)

CPU only.  Run via run_cohort_reference.sh.
  python compute_cohort_stain_reference.py --stain hne
  python compute_cohort_stain_reference.py --stain trichrome
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PREPROC = Path(__file__).resolve().parent
sys.path.insert(0, str(_PREPROC))
import cohort_norm as cn  # noqa: E402

_DATA = Path("/arc/project/st-singha53-1/datasets/ssc")
_CBL = Path("/scratch/st-singha53-1/bhelfert/CBL")
FEAT = _CBL / "output" / "features"
OUT_DIR = _CBL / "output" / "stain_references"

# Canonical per-stain WSI dir + patch coords (stain stats are slide-level, shared
# by every encoder of that stain).
STAIN = {
    "hne": {
        "wsi_dir": _DATA / "hne_10_19_2023",
        "patches": FEAT / "uni_v1_hne/20x_256px_0px_overlap/patches",
        "method": "vahadane",
    },
    "trichrome": {
        "wsi_dir": _DATA / "trichrome_5_31_2024",
        "patches": FEAT / "conch_v1_trichrome/20x_512px_0px_overlap/patches",
        "method": "reinhard",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stain", required=True, choices=list(STAIN))
    ap.add_argument("--n_patches", type=int, default=150,
                    help="patches sampled per slide for stain estimation")
    ap.add_argument("--max_pixels", type=int, default=400_000,
                    help="cap on pooled tissue pixels per slide")
    ap.add_argument("--bg_threshold", type=int, default=220)
    args = ap.parse_args()

    cfg = STAIN[args.stain]
    wsi_dir, pdir, method = cfg["wsi_dir"], cfg["patches"], cfg["method"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slides = sorted(p.stem for p in wsi_dir.glob("*.svs"))
    print(f"[{args.stain}] method={method}  {len(slides)} slides")
    print(f"patches: {pdir}")

    ok_ids = []
    sm_list, mc_list = [], []          # hne (stain matrix, max conc)
    mean_list, std_list = [], []       # trichrome (LAB mean, std)
    t0 = time.time()

    for sid in slides:
        svs = wsi_dir / f"{sid}.svs"
        coords_h5 = pdir / f"{sid}_patches.h5"
        if not coords_h5.exists():
            print(f"  {sid}: no patches h5 — SKIP")
            continue
        px, used = cn.patch_tissue_pixels(coords_h5, svs, n_patches=args.n_patches,
                                          max_pixels=args.max_pixels,
                                          bg_threshold=args.bg_threshold)
        if len(px) < 2000:
            print(f"  {sid}: only {len(px)} tissue px ({used} patches) — SKIP")
            continue

        if method == "vahadane":
            try:
                sm, mc = cn.vahadane_stats(px)
            except Exception as e:
                print(f"  {sid}: vahadane failed: {e} — SKIP")
                continue
            ang = np.degrees(np.arccos(np.clip(
                sm[0] @ sm[1] / np.linalg.norm(sm[0]) / np.linalg.norm(sm[1]), -1, 1)))
            sm_list.append(sm); mc_list.append(mc); ok_ids.append(sid)
            print(f"  {sid}: H={np.round(sm[0],3)} E={np.round(sm[1],3)} "
                  f"angle={ang:.1f}deg maxC={np.round(mc,2)} "
                  f"({len(px)}px/{used}patches)")
        else:
            mean, std = cn.reinhard_stats(px)
            mean_list.append(mean); std_list.append(std); ok_ids.append(sid)
            print(f"  {sid}: LAB mean={np.round(mean,1)} std={np.round(std,1)} "
                  f"({len(px)}px/{used}patches)")

    if not ok_ids:
        sys.exit("No slides produced stain stats.")

    out = OUT_DIR / f"cohort_stain_reference_{args.stain}.npz"
    if method == "vahadane":
        sm_arr = np.array(sm_list)               # (n,2,3)
        mc_arr = np.array(mc_list)               # (n,2)
        tgt_sm = np.median(sm_arr, axis=0)
        tgt_sm = tgt_sm / np.linalg.norm(tgt_sm, axis=1, keepdims=True)
        tgt_mc = np.median(mc_arr, axis=0)
        np.savez(out, stain="hne", method="vahadane", slide_ids=np.array(ok_ids),
                 src_sm=sm_arr, src_mc=mc_arr, target_sm=tgt_sm, target_mc=tgt_mc)
        print(f"\nCohort median target ({len(ok_ids)} slides):")
        print(f"  H={tgt_sm[0]}  E={tgt_sm[1]}  maxC={tgt_mc}")
    else:
        mean_arr = np.array(mean_list)
        std_arr = np.array(std_list)
        tgt_mean = np.median(mean_arr, axis=0)
        tgt_std = np.median(std_arr, axis=0)
        np.savez(out, stain="trichrome", method="reinhard", slide_ids=np.array(ok_ids),
                 src_mean=mean_arr, src_std=std_arr,
                 target_mean=tgt_mean, target_std=tgt_std)
        print(f"\nCohort median target ({len(ok_ids)} slides):")
        print(f"  LAB mean={tgt_mean}  std={tgt_std}")
    print(f"Saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
