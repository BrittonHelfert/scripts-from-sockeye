"""
Stain normalization for histopathology images.

Two normalizers with a shared 3-phase API:

  VahadaneNormalizer — sparse NMF stain decomposition (best for H&E)
  ReinhardNormalizer — LAB color transfer (best for trichrome and multi-stain)

Both follow the same workflow:
    normalizer.fit(reference_rgb)       # once — learn target appearance
    normalizer.fit_source(source_rgb)   # per slide/image — learn source appearance
    normalized = normalizer.transform(patch_rgb)  # per patch/image

Background pixels (grayscale > 200) are always passed through unchanged.

References:
    Vahadane et al. (2016) Structure-Preserving Color Normalization.
    Reinhard et al. (2001) Color Transfer Between Images.
"""

import numpy as np
from sklearn.decomposition import DictionaryLearning
from skimage import color as skcolor

# OD cap: clamp RGB to this minimum before log to avoid extreme OD values
# from ink marks, melanin, etc. RGB=8 -> OD ~3.5
_RGB_MIN = 8
_OD_BETA = 0.15  # minimum OD per channel to be considered stain-bearing
_MAX_PIXELS_DL = 50_000  # subsample limit for dictionary learning


def _rgb_to_od(rgb):
    """Convert RGB (uint8) to optical density, with OD capping."""
    rgb = rgb.astype(np.float64)
    rgb = np.clip(rgb, _RGB_MIN, 255)
    return -np.log(rgb / 255.0)


def _od_to_rgb(od):
    """Convert optical density back to RGB (uint8)."""
    rgb = 255.0 * np.exp(-od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _get_tissue_mask(img_rgb, threshold=200):
    """Simple tissue mask: pixels darker than threshold in grayscale."""
    gray = np.mean(img_rgb, axis=2)
    return gray < threshold


def _get_stain_matrix(tissue_rgb, n_stains=2, alpha=0.1, rng_seed=42):
    """Extract stain matrix from tissue pixels using dictionary learning.

    Returns a (n_stains, 3) matrix where each row is a normalized stain
    vector in OD space.
    """
    pixels = tissue_rgb.reshape(-1, 3)
    od = _rgb_to_od(pixels)

    # Keep only stain-bearing pixels (all OD channels above threshold)
    mask = np.all(od > _OD_BETA, axis=1)
    od_filtered = od[mask]

    if od_filtered.shape[0] < 10:
        # Fallback: standard H&E stain vectors
        return np.array([[0.644, 0.717, 0.267],
                         [0.093, 0.954, 0.283]])

    # Subsample for speed
    if od_filtered.shape[0] > _MAX_PIXELS_DL:
        rng = np.random.RandomState(rng_seed)
        idx = rng.choice(od_filtered.shape[0], _MAX_PIXELS_DL, replace=False)
        od_filtered = od_filtered[idx]

    dl = DictionaryLearning(
        n_components=n_stains,
        alpha=alpha,
        transform_algorithm="lasso_lars",
        positive_dict=True,
        positive_code=True,
        max_iter=200,
        random_state=rng_seed,
        fit_algorithm="cd",
    )
    dl.fit(od_filtered)
    stain_matrix = dl.components_

    # Normalize rows
    norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    stain_matrix /= norms

    # Convention: first row = hematoxylin (higher red OD component)
    if stain_matrix[0, 0] < stain_matrix[1, 0]:
        stain_matrix = stain_matrix[[1, 0]]

    return stain_matrix


def _get_concentrations(rgb, stain_matrix):
    """Get stain concentrations via least-squares, clipped to non-negative."""
    pixels = rgb.reshape(-1, 3)
    od = _rgb_to_od(pixels)
    conc = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
    return np.clip(conc, 0, None)


class VahadaneNormalizer:
    """Vahadane stain normalizer for H&E patches.

    Workflow:
        1. fit(reference_rgb) — learn target stain appearance (once)
        2. fit_source(source_thumbnail_rgb) — learn source stain appearance (per slide)
        3. transform(patch_rgb) — normalize a patch (per patch)
    """

    def __init__(self):
        self._target_stain = None
        self._target_max_conc = None
        self._src_stain = None
        self._src_max_conc = None

    def fit(self, target_rgb):
        """Learn target stain matrix and concentration scale from reference image.

        Args:
            target_rgb: Reference image as (H, W, 3) uint8 numpy array.
                        Should contain tissue (will be auto-masked).
        """
        mask = _get_tissue_mask(target_rgb)
        tissue = target_rgb[mask]
        self._target_stain = _get_stain_matrix(tissue)
        conc = _get_concentrations(tissue, self._target_stain)
        self._target_max_conc = np.percentile(conc, 99, axis=0)
        self._target_max_conc[self._target_max_conc == 0] = 1.0

    @property
    def source_fitted(self):
        """True if fit_source() completed successfully."""
        return self._src_stain is not None

    def fit_source(self, source_rgb):
        """Learn source stain matrix and concentration scale from a slide thumbnail.

        Args:
            source_rgb: Slide thumbnail as (H, W, 3) uint8 numpy array.
                        Should contain tissue (will be auto-masked).
        """
        mask = _get_tissue_mask(source_rgb)
        tissue = source_rgb[mask]
        if tissue.shape[0] < 10:
            self._src_stain = None
            return
        self._src_stain = _get_stain_matrix(tissue)
        conc = _get_concentrations(tissue, self._src_stain)
        self._src_max_conc = np.percentile(conc, 99, axis=0)
        self._src_max_conc[self._src_max_conc == 0] = 1.0

    def transform(self, patch_rgb):
        """Normalize a tissue patch using pre-computed source and target stats.

        Background pixels (grayscale > 200) are passed through unchanged
        to avoid pink tinting from OD decomposition noise.

        Args:
            patch_rgb: Tissue patch as (H, W, 3) uint8 numpy array.

        Returns:
            Normalized patch as (H, W, 3) uint8 numpy array.
            Returns input unchanged if source fitting failed.
        """
        if self._src_stain is None or self._target_stain is None:
            return patch_rgb

        h, w, _ = patch_rgb.shape
        mask = _get_tissue_mask(patch_rgb)

        # If no tissue pixels, return unchanged
        if not mask.any():
            return patch_rgb

        result = patch_rgb.copy()
        tissue = patch_rgb[mask]

        # Decompose into source stain concentrations
        conc = _get_concentrations(tissue, self._src_stain)

        # Scale concentrations to match target intensity
        conc *= (self._target_max_conc / self._src_max_conc)

        # Reconstruct with target stain vectors
        od = conc @ self._target_stain
        result[mask] = _od_to_rgb(od)

        return result


class ReinhardNormalizer:
    """Reinhard stain normalizer for multi-stain images (trichrome, etc.).

    Matches LAB color statistics between source and target images using
    only tissue pixels. Unlike Vahadane, makes no assumptions about the
    number of stains — works for trichrome (3 stains) and other non-H&E.

    Workflow:
        1. fit(reference_rgb) — learn target LAB stats (once)
        2. fit_source(source_rgb) — learn source LAB stats (per slide/image)
        3. transform(patch_rgb) — normalize a patch (per patch/image)
    """

    def __init__(self):
        self._target_mean = None  # (3,) LAB channel means on tissue
        self._target_std = None   # (3,) LAB channel stds on tissue
        self._src_mean = None
        self._src_std = None

    def fit(self, target_rgb):
        """Learn target LAB statistics from a reference image.

        Args:
            target_rgb: Reference image as (H, W, 3) uint8 numpy array.
                        Background is auto-masked; only tissue pixels used.
        """
        mask = _get_tissue_mask(target_rgb)
        tissue = target_rgb[mask]
        if tissue.shape[0] < 10:
            raise ValueError("Reference image has too few tissue pixels to fit.")
        lab = skcolor.rgb2lab(tissue.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        self._target_mean = lab.mean(axis=0)
        self._target_std = lab.std(axis=0)

    @property
    def source_fitted(self):
        """True if fit_source() completed successfully."""
        return self._src_mean is not None

    def fit_source(self, source_rgb):
        """Learn source LAB statistics from a slide thumbnail or image.

        Args:
            source_rgb: Source image as (H, W, 3) uint8 numpy array.
                        Background is auto-masked; only tissue pixels used.
                        Sets src stats to None if too few tissue pixels.
        """
        mask = _get_tissue_mask(source_rgb)
        tissue = source_rgb[mask]
        if tissue.shape[0] < 10:
            self._src_mean = None
            self._src_std = None
            return
        lab = skcolor.rgb2lab(tissue.reshape(1, -1, 3) / 255.0).reshape(-1, 3)
        self._src_mean = lab.mean(axis=0)
        self._src_std = lab.std(axis=0)
        # Guard against zero std (uniform channel — no variation to transfer)
        self._src_std = np.where(self._src_std < 1e-6, 1.0, self._src_std)

    def transform(self, patch_rgb):
        """Normalize a patch using pre-computed source and target LAB stats.

        Background pixels (grayscale > 200) are passed through unchanged.

        Args:
            patch_rgb: Image patch as (H, W, 3) uint8 numpy array.

        Returns:
            Normalized patch as (H, W, 3) uint8 numpy array.
            Returns input unchanged if source fitting failed.
        """
        if self._src_mean is None or self._target_mean is None:
            return patch_rgb

        mask = _get_tissue_mask(patch_rgb)
        if not mask.any():
            return patch_rgb

        # Convert full image to LAB (float [0,1] input for skimage)
        lab = skcolor.rgb2lab(patch_rgb.astype(np.float64) / 255.0)

        # Apply per-channel transfer on tissue pixels only
        for ch in range(3):
            lab[mask, ch] = (
                (lab[mask, ch] - self._src_mean[ch]) / self._src_std[ch]
                * self._target_std[ch] + self._target_mean[ch]
            )

        # Convert back to uint8 RGB; background pixels are untouched in lab
        # but we restore them explicitly to avoid any float conversion drift
        rgb_norm = (np.clip(skcolor.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
        rgb_norm[~mask] = patch_rgb[~mask]

        return rgb_norm
