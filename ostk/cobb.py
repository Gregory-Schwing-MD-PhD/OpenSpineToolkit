"""ostk.cobb — coronal Cobb angle (scoliosis) from a v3 label volume.

Companion to ostk.metrics' sagittal pipeline (PI/LL): the per-vertebra
endplate normal comes from the same ostk.spine.fit_endplate primitive
(anterior-filtered, tilt-aware), and the viewing axis is derived from the
same femoral-head L-R estimate (ostk.metrics._lr_axis_from_label) — but
rotated in-plane to the anterior axis (ostk.spine.anterior_axis) rather than
used directly, since Cobb/scoliosis is measured in the CORONAL view while
LL/lordosis is measured in the SAGITTAL view (see geometry.cobb_angle's own
docstring: view_normal=L-R -> sagittal; view_normal=A-P -> coronal).

Reuses _lr_axis_from_label rather than re-deriving the L-R axis, so the
coronal and sagittal pipelines share one patient-axis estimate per case.

Pair selection mirrors the clinical convention used in earlier
(pre-toolbox) Cobb scripts: prefer the default full lumbosacral span
(L1->S1, falling back to L1->L5, L1->L4, L2->S1, L2->L5 by availability);
only switch to a non-default, non-adjacent pair if it beats the default by
more than _OVERRIDE_MARGIN_DEG, so a noisy short segment can't override the
clinically meaningful full-span measurement.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, cobb_angle
from .spine import anterior_axis, fit_endplate
from .metrics import _lr_axis_from_label
from .record import Measurement

COBB_METHOD_VERSION = "cobb-v1"

# Cranial -> caudal candidate levels for the lumbar/lumbosacral curve.
COBB_LEVEL_CHAIN = ("L1", "L2", "L3", "L4", "L5", "L6", "S1")

_DEFAULT_PAIR_CANDIDATES = (("L1", "S1"), ("L1", "L5"), ("L1", "L4"),
                           ("L2", "S1"), ("L2", "L5"))
_MIN_NONADJACENT_SEP = 2
_OVERRIDE_MARGIN_DEG = 3.0
_FIT_RESIDUAL_FLAG_MM = 2.0  # not yet validated -- tune once manual-measurement
                             # comparison (SPEC §7) gives a real residual distribution


def _bottom_endplate_which(level: str) -> str:
    """S1's clinically meaningful Cobb landmark is its SUPERIOR endplate (the
    lumbosacral junction), matching ostk.metrics.LL_ENDPLATE_CHAIN's
    convention -- not its inferior surface."""
    return "superior" if level == "S1" else "inferior"


def _cranial_normal(fit: Optional[Tuple[np.ndarray, np.ndarray, float]],
                    which: str) -> Optional[np.ndarray]:
    """fit_endplate orients 'superior' normals cranially but 'inferior'
    normals CAUDALLY (opposite sign convention -- confirmed against the real
    source, not assumed). cobb_angle needs both endplate normals in one
    consistent orientation to avoid measuring the supplementary (~180-x)
    angle instead of the true tilt difference, so flip 'inferior' normals
    back to cranial here before any Cobb comparison. This mirrors the
    convention the earlier (pre-ostk) Cobb scripts used uniformly."""
    if fit is None:
        return None
    n = fit[1]
    return -n if which == "inferior" else n


def coronal_cobb_from_label(label, affine, *, case_id: str = "",
                            sup_axis=WORLD_SUPERIOR, ant_frac: float = 0.6,
                            head_frac: float = 0.35, min_voxels: int = 30
                            ) -> Measurement:
    """Auto-detect the most-tilted vertebral pair over the lumbar + S1 levels
    and return the coronal Cobb angle as a Measurement (SPEC §4 contract).
    Never silently drops a bad case -- returns value=None with qc_flags set
    if no usable pair is found."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world

    flags: List[str] = []
    lr, lr_ok = _lr_axis_from_label(label, affine, sup_axis, head_frac, min_voxels)
    if not lr_ok:
        flags.append("sagittal_ref_fallback")
    ap = anterior_axis(sup_axis, lr)

    fits: Dict[str, Dict[str, Optional[Tuple[np.ndarray, np.ndarray, float]]]] = {}
    landmarks: Dict[str, object] = {}
    residuals: Dict[str, float] = {}
    for level in COBB_LEVEL_CHAIN:
        try:
            vid = lid(level)
        except KeyError:
            continue
        m = binary_mask(label, vid)
        if level == "S1" and not m.any():
            m = binary_mask(label, lid("sacrum"))
        pts = mask_world(largest_component(m), affine)
        sup = fit_endplate(pts, sup_axis, "superior", ant_frac, min_points=min_voxels)
        inf = fit_endplate(pts, sup_axis, "inferior", ant_frac, min_points=min_voxels)
        if sup is None and inf is None:
            continue
        fits[level] = {"superior": sup, "inferior": inf}
        if sup is not None:
            landmarks[f"{level}_superior_endplate"] = sup[0].tolist()
            residuals[f"{level}_superior_rms"] = round(sup[2], 3)
        if inf is not None:
            landmarks[f"{level}_inferior_endplate"] = inf[0].tolist()
            residuals[f"{level}_inferior_rms"] = round(inf[2], 3)

    avail = [lv for lv in COBB_LEVEL_CHAIN if lv in fits]
    if not avail:
        flags.append("no_levels_available")
        return Measurement(case_id=case_id, parameter="coronal_cobb_angle",
                           value=None, qc_flags=flags or ["ok"],
                           method_version=COBB_METHOD_VERSION, supine_ct=True)

    def pair_angle(top: str, bot: str) -> float:
        top_fit = fits[top]["superior"]
        bot_which = _bottom_endplate_which(bot)
        bot_fit = fits[bot][bot_which]
        n_top = _cranial_normal(top_fit, "superior")
        n_bot = _cranial_normal(bot_fit, bot_which)
        if n_top is None or n_bot is None:
            return float("nan")
        return cobb_angle(n_top, n_bot, ap)

    default_pair: Tuple[Optional[str], Optional[str]] = (None, None)
    default_angle = float("nan")
    for top_c, bot_c in _DEFAULT_PAIR_CANDIDATES:
        if top_c in fits and bot_c in fits:
            default_pair = (top_c, bot_c)
            default_angle = pair_angle(top_c, bot_c)
            break

    best_alt_pair: Tuple[Optional[str], Optional[str]] = (None, None)
    best_alt_angle = float("nan")
    for i, top in enumerate(avail):
        for j, bot in enumerate(avail):
            if j <= i + _MIN_NONADJACENT_SEP - 1:
                continue
            if (top, bot) == default_pair:
                continue
            a = pair_angle(top, bot)
            if not math.isnan(a) and (math.isnan(best_alt_angle) or a > best_alt_angle):
                best_alt_angle, best_alt_pair = a, (top, bot)

    if (not math.isnan(best_alt_angle) and not math.isnan(default_angle)
            and best_alt_angle > default_angle + _OVERRIDE_MARGIN_DEG):
        chosen_pair, chosen_angle = best_alt_pair, best_alt_angle
    elif not math.isnan(default_angle):
        chosen_pair, chosen_angle = default_pair, default_angle
    else:
        chosen_pair, chosen_angle = best_alt_pair, best_alt_angle

    top_lv, bot_lv = chosen_pair
    if top_lv is None or bot_lv is None:
        flags.append("no_valid_pair")
        return Measurement(case_id=case_id, parameter="coronal_cobb_angle",
                           value=None, qc_flags=flags or ["ok"],
                           method_version=COBB_METHOD_VERSION, supine_ct=True)

    for lv, which in ((top_lv, "superior"), (bot_lv, _bottom_endplate_which(bot_lv))):
        fit = fits[lv][which]
        if fit is not None and fit[2] > _FIT_RESIDUAL_FLAG_MM:
            flags.append(f"fit_residual_high:{lv}_{which}")

    landmarks["cobb_top_level"] = top_lv
    landmarks["cobb_bottom_level"] = bot_lv
    return Measurement(
        case_id=case_id, parameter="coronal_cobb_angle",
        value=round(chosen_angle, 3),
        landmarks_world_mm=landmarks, fit_residuals=residuals,
        qc_flags=flags or ["ok"], method_version=COBB_METHOD_VERSION,
        supine_ct=True)
