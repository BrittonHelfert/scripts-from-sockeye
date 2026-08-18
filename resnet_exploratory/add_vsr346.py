#!/usr/bin/env python3
"""Bring vsr346 back into the cohort as dcSSc-noILD, with CORRECTED embeddings.

vsr346 is the 3rd tissue piece on H&E slide 261317 / trichrome slide 269311.
The user labelled it diffuse (dcSSc) no-ILD. It was added once on 2026-06-11 but
the matrices were later rebuilt without it. This re-adds it everywhere:

  labels.csv          : append  vsr346, outcome=SSc-noILD
  sample_subsets.csv  : append  vsr346, subset=dcSSc
  embeddings (x8)      : append vsr346 row pooled via contour containment
                        (common.contour_assign -- the SAME corrected method that
                        re-pooled vsr341/vsr344, NOT the old x-gap split that mixed
                        vsr344<->vsr346). Verifies slide-mates reproduce the
                        existing (corrected) matrix rows before trusting vsr346.
  interp (x3)          : append from output/histolytics_features/{hne,hne_norm,
                        trichrome}/vsr346_features.csv (already correct)
  tabular (x3)         : append clinical/cytokines/autoantibodies from the
                        _vsr346_*_row.csv helpers (run extract_vsr346_xlsx.py first)

NON-DESTRUCTIVE + idempotent: existing rows/cols preserved, vsr346 only appended
if absent; originals backed up to feature_matrices/_pre_vsr346/.

Run on a compute node (CPU only).  python add_vsr346.py [--dry-run]
"""
from __future__ import annotations

import sys
import shutil
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
HIS = _CBL / "output" / "histolytics_features"
BACKUP = MAT / "_pre_vsr346"
MAP_CSV = _DATA / "he_tri_image_mapping.csv"

TARGET = "vsr346"
OUTCOME = "SSc-noILD"      # diffuse no-ILD; binary label 0 via run_cv LABEL_MAP
SUBSET = "dcSSc"
HE_SLIDE, TRI_SLIDE = 261317, 269311
REPRO_TOL = 1e-4

# matrix -> (features h5 dir, contours dir, slide)
EMBED = {
    "uni_hne":         (FEAT / "uni_v1_hne/20x_256px_0px_overlap/features_uni_v1",            FEAT / "uni_v1_hne/contours_geojson",            HE_SLIDE),
    "uni_hne_norm":    (FEAT / "uni_v1_hne/20x_256px_0px_overlap/features_uni_v1_vahadane",   FEAT / "uni_v1_hne/contours_geojson",            HE_SLIDE),
    "resnet_hne":      (FEAT / "resnet50_hne_20x/20x_256px_0px_overlap/features_resnet50",          FEAT / "resnet50_hne_20x/contours_geojson",      HE_SLIDE),
    "resnet_hne_norm": (FEAT / "resnet50_hne_20x/20x_256px_0px_overlap/features_resnet50_vahadane", FEAT / "resnet50_hne_20x/contours_geojson",      HE_SLIDE),
    "conch_tri":       (FEAT / "conch_v1_trichrome/20x_512px_0px_overlap/features_conch_v1",          FEAT / "conch_v1_trichrome/contours_geojson",    TRI_SLIDE),
    "conch_tri_norm":  (FEAT / "conch_v1_trichrome/20x_512px_0px_overlap/features_conch_v1_reinhard", FEAT / "conch_v1_trichrome/contours_geojson",    TRI_SLIDE),
    "resnet_tri":      (FEAT / "resnet50_trichrome_20x/20x_256px_0px_overlap/features_resnet50",          FEAT / "resnet50_trichrome_20x/contours_geojson", TRI_SLIDE),
    "resnet_tri_norm": (FEAT / "resnet50_trichrome_20x/20x_256px_0px_overlap/features_resnet50_reinhard", FEAT / "resnet50_trichrome_20x/contours_geojson", TRI_SLIDE),
}
INTERP = {"interp_hne": HIS / "hne" / f"{TARGET}_features.csv",
          "interp_hne_norm": HIS / "hne_norm" / f"{TARGET}_features.csv",
          "interp_tri": HIS / "trichrome" / f"{TARGET}_features.csv"}
TABULAR = {"clinical": MAT / "_vsr346_clinical_row.csv",
           "cytokines": MAT / "_vsr346_cytokine_row.csv",
           "autoantibodies": MAT / "_vsr346_aa_row.csv"}


def _backup(path: Path):
    BACKUP.mkdir(parents=True, exist_ok=True)
    dst = BACKUP / path.name
    if not dst.exists():
        shutil.copy2(path, dst)


def vsr_ids_for_slide(slide_id):
    m = pd.read_csv(MAP_CSV)
    m.columns = [c.strip().lstrip("﻿") for c in m.columns]
    for col in ("he", "trichrome"):
        hit = m[m[col].astype("Int64") == slide_id]
        if len(hit):
            r = hit.iloc[0]
            return [str(r[c]).strip() for c in ("sub1", "sub2", "sub3", "sub4")
                    if pd.notna(r[c]) and str(r[c]).strip().upper() != "NA"]
    raise RuntimeError(f"slide {slide_id} not in mapping")


def _append_row(name, row: pd.Series, dry):
    """Append a sample row (index=feature cols, name=TARGET) to MAT/{name}.csv."""
    p = MAT / f"{name}.csv"
    if not p.exists():
        print(f"  [{name}] SKIP: {p} missing")
        return
    mat = pd.read_csv(p)
    idc = mat.columns[0]
    if TARGET in set(mat[idc]):
        print(f"  [{name}] already has {TARGET} (n={len(mat)}) — untouched")
        return
    fcols = [c for c in mat.columns if c != idc]
    aligned = row.reindex(fcols)
    miss = aligned[aligned.isna()].index.tolist()
    if miss:
        print(f"  [{name}] WARN {len(miss)} matrix cols absent from {TARGET} row "
              f"(left NaN): {miss[:5]}{'...' if len(miss) > 5 else ''}")
    out = pd.concat([mat, pd.DataFrame([{idc: TARGET, **aligned.to_dict()}])],
                    ignore_index=True).sort_values(idc).reset_index(drop=True)
    if dry:
        print(f"  [{name}] DRY would write n={len(out)} (was {len(mat)})")
        return
    _backup(p)
    out.to_csv(p, index=False)
    print(f"  [{name}] wrote n={len(out)} (was {len(mat)})  [backup -> _pre_vsr346/]")


def do_labels(dry):
    print("\n== labels.csv / sample_subsets.csv ==")
    lp = MAT / "labels.csv"
    lab = pd.read_csv(lp)
    if TARGET in set(lab["sample_id"]):
        print(f"  labels.csv already has {TARGET}")
    else:
        out = pd.concat([lab, pd.DataFrame([{"sample_id": TARGET, "outcome": OUTCOME}])],
                        ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        if dry:
            print(f"  labels.csv DRY would add {TARGET}={OUTCOME} (n {len(lab)}->{len(out)})")
        else:
            _backup(lp); out.to_csv(lp, index=False)
            print(f"  labels.csv added {TARGET}={OUTCOME} (n={len(out)})")
    sp = MAT / "sample_subsets.csv"
    sub = pd.read_csv(sp)
    if TARGET in set(sub["sample_id"]):
        print(f"  sample_subsets.csv already has {TARGET}")
    else:
        out = pd.concat([sub, pd.DataFrame([{"sample_id": TARGET, "subset": SUBSET}])],
                        ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        if dry:
            print(f"  sample_subsets.csv DRY would add {TARGET}={SUBSET} (n {len(sub)}->{len(out)})")
        else:
            _backup(sp); out.to_csv(sp, index=False)
            print(f"  sample_subsets.csv added {TARGET}={SUBSET} (n={len(out)})")


def do_embeddings(dry):
    print("\n== Embedding matrices (contour-containment pooling) ==")
    for name, (fdir, cdir, slide) in EMBED.items():
        h5 = fdir / f"{slide}.h5"
        gj = cdir / f"{slide}.geojson"
        if not h5.exists() or not gj.exists():
            print(f"  [{name}] SKIP: missing h5/geojson")
            continue
        vsr_ids = vsr_ids_for_slide(slide)
        with h5py.File(h5, "r") as f:
            coords = f["coords"][:]
            patch_l0 = float(f["coords"].attrs["patch_size_level0"])
            feats = f["features"][:]
        assign, _, info = contour_assign.assign_patches(coords, patch_l0, gj, vsr_ids)
        if info["status"] != "ok":
            print(f"  [{name}] SKIP: assign {info['status']}")
            continue
        idx = assign.get(TARGET)
        if idx is None or idx.size == 0:
            print(f"  [{name}] SKIP: {TARGET} got 0 patches")
            continue
        # verify slide-mates reproduce the (corrected) matrix rows
        mat = pd.read_csv(MAT / f"{name}.csv")
        idc = mat.columns[0]; fcols = [c for c in mat.columns if c != idc]
        checks = []
        for sub in vsr_ids:
            if sub == TARGET or sub not in set(mat[idc]) or assign[sub].size == 0:
                continue
            exist = mat.loc[mat[idc] == sub, fcols].values[0].astype(float)
            checks.append((sub, float(np.max(np.abs(exist - feats[assign[sub]].mean(0))))))
        status = "OK" if checks and all(d < REPRO_TOL for _, d in checks) else "??"
        print(f"  [{name}] {TARGET}: {idx.size} patches | slide-mate repro {status}: "
              + ", ".join(f"{s} Δ={d:.1e}" for s, d in checks))
        vec = feats[idx].mean(0)
        _append_row(name, pd.Series(vec, index=[f"f{i}" for i in range(len(vec))], name=TARGET), dry)


def do_interp(dry):
    print("\n== Interp matrices ==")
    for name, csv in INTERP.items():
        if not csv.exists():
            print(f"  [{name}] SKIP: {csv} missing")
            continue
        s = pd.read_csv(csv).iloc[0]
        if "sample_id" in s.index:
            s = s.drop(labels=["sample_id"])
        _append_row(name, pd.Series(s.values, index=s.index, name=TARGET), dry)


def do_tabular(dry):
    print("\n== Tabular matrices ==")
    for name, csv in TABULAR.items():
        if not csv.exists():
            print(f"  [{name}] SKIP: helper {csv.name} missing "
                  f"(run extract_vsr346_xlsx.py first)")
            continue
        s = pd.read_csv(csv).iloc[0]
        if "sample_id" in s.index:
            s = s.drop(labels=["sample_id"])
        row = pd.Series(s.values, index=s.index, name=TARGET)
        if "outcome" in pd.read_csv(MAT / f"{name}.csv").columns:
            row["outcome"] = OUTCOME
        _append_row(name, row, dry)


def main():
    dry = "--dry-run" in sys.argv
    print(f"Re-adding {TARGET} ({OUTCOME}, {SUBSET}) {'[DRY-RUN]' if dry else '[WRITE]'}")
    do_labels(dry)
    do_embeddings(dry)
    do_interp(dry)
    do_tabular(dry)
    print("\nDone.")


if __name__ == "__main__":
    main()
