# Cohort-median stain normalization — method notes (2026-06-18)

New normalized-embedding run. The requested recipe was **Macenko for H&E**,
**Reinhard for trichrome**, normalizing every slide to the **cohort-median** stain
statistics, using **tissue patches only**, then **L2-normalizing each patch
embedding before mean-pooling**.

## H&E: Macenko was degenerate → switched to Vahadane

Macenko, implemented exactly as in `biomni_scripts/macenko_median_normalize.py`,
**fails on this cohort**. Diagnosis (per-slide, all 18 H&E slides, tested at both
the whole-slide thumbnail resolution AND full level-0 patch resolution):

- The two estimated stain vectors come out **near-collinear**: H/E angle ≈
  **0.5–2°** on every slide (a healthy H&E pair is ~15–40° apart).
- Cause is the tissue, not a code bug: skin dermis is collagen/eosin-dominated
  with sparse, weak hematoxylin, so the OD cloud has almost no hue spread on the
  hematoxylin axis. Macenko's percentile-angle method then can't separate the
  stains.
- Consequence: the 2-stain projection is rank-deficient, so normalization
  **collapses patches to gray** (no concentration clipping) or **over-darkens
  them** (with clipping). Verified same-slide round-trip (source==target, which
  should be ~identity): `[203,172,185] → [217,217,217]` (gray) or `→ [57,57,57]`
  (dark). Either output would corrupt the embeddings.

**Vahadane** (sparse-NMF stain decomposition — the method behind the project's
prior headline result) recovers well-separated H/E (angle ≈ **17°**) and round-
trips cleanly (`[194,148,165] → [183,139,160]`, hue preserved, no NaN). So H&E is
normalized with **Vahadane to the cohort-median stain matrix + max-concentration**.
This keeps the user's actual intent — a *median stain-vector* reference estimated
from tissue only — with a stain-separation method that works on this data.

The degenerate Macenko `.npz` references are kept under
`stain_references/_macenko_degenerate/` for reference; they are NOT used.

## Trichrome: Reinhard (as requested)

Reinhard needs no stain separation (it matches LAB channel mean/std), so it is
robust here. Trichrome uses **Reinhard to the cohort-median LAB mean/std**.

## Pipeline summary

| Stage | Script | Output |
|-------|--------|--------|
| 1. cohort reference | `preprocessing/compute_cohort_stain_reference.py` | `cohort_stain_reference_{hne,trichrome}.npz` (per-slide source stats + cohort-median target) |
| 2. normed embeddings | `preprocessing/extract_cohort_normed_features.py` | `features_<enc>_vahadane_med/` (H&E), `features_<enc>_reinhardmed/` (tri) |
| 3. feature matrices | `preprocessing/build_normed_matrices.py` | `feature_matrices/{resnet_hne,uni_hne,resnet_tri,conch_tri}_mnorm.csv` |

- Stain stats are estimated from **TRIDENT tissue patches** at full (level-0)
  resolution, background masked — never the mostly-blank whole slide.
- Each patch embedding is **L2-normalized before mean-pooling** per sample, so
  high-norm patches can't dominate the sample mean.
- Sample assignment uses **contour containment** (`common.contour_assign`), the
  corrected method (handles the vsr344/vsr346 x-overlap, drops debris/off-tissue
  patches).
