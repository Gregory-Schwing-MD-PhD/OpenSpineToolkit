"""ostk.surgery — synthesise a patient's spine AFTER a lordosis-restoring operation.

Every lordosing op in Greenberg Ch.73 reduces to the same move — rotate the spinal
segment cranial to the operative level by Δ° in the sagittal plane (pelvis fixed) — and
they differ only by the FULCRUM and the tissue change at the hinge:

  interbody / ALIF / LLIF / TLIF / ACR : posterior fulcrum, disc OPENS anteriorly,
                                         a cage fills the opened disc.
  SPO (Smith-Petersen)                 : mid-disc fulcrum (posterior elements resected,
                                         not modelled on body-only labels).
  PSO (pedicle subtraction)            : anterior-cortex fulcrum, a body wedge is
                                         RESECTED and CLOSED (the column shortens).

Phase 1: the rigid rotation (angle is fulcrum-independent — validated by re-measuring).
Phase 2 (here): the technique-correct fulcrum (so the upper spine TRANSLATES correctly,
which drives SVA / global balance) + cage insertion vs body-wedge resection at the hinge.
Operates on the per-vertebra LABEL volume; CT-intensity realism is Phase 3.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometry import (WORLD_SUPERIOR, rotation_matrix, unit, cobb_angle,
                       project_out, angle_between)
from .labels import LABELS

# Cranial → caudal vertebral chain. The "mobile" segment for a correction at `level`
# is `level` and everything ABOVE it; S1/sacrum/femurs are never mobile (pelvic anchor).
SPINE_CRANIOCAUDAL: List[str] = (
    [f"T{n}" for n in range(1, 14)] + ["L1", "L2", "L3", "L4", "L5", "L6"]
)
_FULL_CHAIN = SPINE_CRANIOCAUDAL + ["S1"]            # incl. S1 for "vertebra below"

CAGE_ID = 70                                         # synthetic interbody-cage label

# technique -> (fulcrum position at the hinge, reconciliation mode)
TECHNIQUES = {
    "alif": ("posterior", "cage"),
    "llif": ("posterior", "cage"),
    "tlif": ("posterior", "cage"),
    "interbody": ("posterior", "cage"),
    "acr":  ("posterior", "cage"),
    "spo":  ("mid", None),
    "pso":  ("anterior", "resect"),
}


def mobile_ids_for_level(level: str, present_ids) -> List[int]:
    """Label ids of the vertebrae at or cranial to `level` (the segment a correction
    at `level` swings). `level` is the lowest MOBILE vertebra — e.g. an L5–S1 ALIF is
    level='L5' (L5 and up move, S1 stays). S1/sacrum are never included."""
    if level not in SPINE_CRANIOCAUDAL:
        raise ValueError(f"level {level!r} must be one of {SPINE_CRANIOCAUDAL}")
    names = SPINE_CRANIOCAUDAL[: SPINE_CRANIOCAUDAL.index(level) + 1]
    pres = set(int(v) for v in present_ids)
    return [LABELS[n] for n in names if LABELS[n] in pres]


def _vertebra_below(level: str) -> Optional[str]:
    i = _FULL_CHAIN.index(level)
    return _FULL_CHAIN[i + 1] if i + 1 < len(_FULL_CHAIN) else None


def _lr_axis(label, affine, sup_axis) -> np.ndarray:
    try:
        from .metrics import femoral_head_center
        L = femoral_head_center(label, affine, "femur_left", "left_hip", sup_axis=sup_axis)
        R = femoral_head_center(label, affine, "femur_right", "right_hip", sup_axis=sup_axis)
        if L is not None and R is not None:
            return unit(R[0] - L[0])
    except Exception:
        pass
    return unit(np.array([1.0, 0.0, 0.0]))


def _hinge_fulcrum(label, affine, level, position, sup_axis, lr) -> Optional[np.ndarray]:
    """Fulcrum at the operative disc/level: the anterior or posterior corner (or mid)
    of `level`'s INFERIOR endplate. Returns None if the corners can't be found (the
    caller falls back to the level centroid; the angle is unchanged either way)."""
    try:
        from .spine import endplate_corners, corner_params_for_level
        from .masks import binary_mask, largest_component, mask_world
        pts = mask_world(largest_component(binary_mask(label, LABELS[level])), affine)
        A_c, P_c, _ = endplate_corners(pts, normal_axis=sup_axis, which="inferior",
                                       lr=lr, **corner_params_for_level(level))
    except Exception:
        return None
    if position == "anterior":
        return np.asarray(A_c, float)
    if position == "posterior":
        return np.asarray(P_c, float)
    return 0.5 * (np.asarray(A_c, float) + np.asarray(P_c, float))


def _oriented_theta(label, affine, level, delta_deg, lr, sup_axis) -> float:
    th = float(np.deg2rad(abs(delta_deg)))
    try:
        from .spine import endplate_from_label
        _, n_lvl, _ = endplate_from_label(label, affine, level, "superior", normal_axis=sup_axis)
        _, n_s1, _ = endplate_from_label(label, affine, "S1", "superior", normal_axis=sup_axis)
    except Exception:
        return th
    plus = cobb_angle(rotation_matrix(lr, th) @ n_lvl, n_s1, lr)
    minus = cobb_angle(rotation_matrix(lr, -th) @ n_lvl, n_s1, lr)
    return th if plus >= minus else -th


def _si_axis_and_sign(affine, sup_axis):
    """(voxel axis most parallel to the superior direction, +1 if increasing that index
    goes cranial)."""
    M = np.asarray(affine, float)[:3, :3]
    proj = (M / (np.linalg.norm(M, axis=0) + 1e-9)).T @ unit(sup_axis)
    k = int(np.argmax(np.abs(proj)))
    return k, (proj[k] >= 0)


def _fill_disc_cage(out, rot_level_mask, below_mask, cage_id, k, cranial_is_plus):
    """Fill the disc gap OPENED between the rotated operative vertebra and the fixed
    vertebra below it with a cage label, per (in-plane) column along the SI axis."""
    o = np.moveaxis(out, k, -1)
    lvl = np.moveaxis(rot_level_mask, k, -1)
    bel = np.moveaxis(below_mask, k, -1)
    if not cranial_is_plus:                          # orient so +index = cranial
        o, lvl, bel = o[..., ::-1], lvl[..., ::-1], bel[..., ::-1]
    nz = o.shape[-1]
    idx = np.arange(nz)
    cols = lvl.any(-1) & bel.any(-1)                 # the disc footprint
    bel_top = np.where(bel, idx, -1).max(-1)         # caudal vertebra's cranial face
    lvl_bot = np.where(lvl, idx, nz).min(-1)         # operative vertebra's caudal face
    gap = (cols[..., None] & (idx > bel_top[..., None]) & (idx < lvl_bot[..., None])
           & (o == 0))
    o[gap] = cage_id                                 # writes through the view -> `out`


def predict_compensated_alignment(pi: float, pt: float, *, target_pt: float = 20.0):
    """Phase 2.5 (analytic) — the post-op STANDING angles after the pelvis releases its
    compensatory retroversion. Re-posturing about the hips is rigid, so PI (and LL) are
    invariant; only the gravity-referenced angles change, and PT+SS=PI always. The
    pelvis anteverts until PT reaches `target_pt` (no change if already ≤ target):

        PT_post = min(pt, target_pt),  SS_post = pi − PT_post,  rotation = pt − PT_post.

    Exact (no voxel resampling); use this for the predicted standing PT/SS in the report.
    The voxel realisation for the synthetic IMAGE is compensate_pelvis (Phase 3)."""
    pt_post = min(pt, target_pt)
    return {"PT": round(pt_post, 3),
            "SS": round(pi - pt_post, 3),
            "pelvic_rotation_deg": round(pt - pt_post, 3)}


def _rotate_ids(label, affine, ids, F, lr, theta):
    """Return a copy of `label` with voxels of `ids` rigidly rotated by `theta` rad
    about world point `F`, axis `lr` (rotated voxels overwrite at overlaps)."""
    from scipy import ndimage
    label = np.asarray(label)
    A = np.asarray(affine, float)
    Rinv = rotation_matrix(lr, -theta)                 # affine_transform pulls (output->input)
    Tn = np.eye(4)
    Tn[:3, :3] = Rinv
    Tn[:3, 3] = F - Rinv @ F
    M = np.linalg.inv(A) @ Tn @ A
    seg = np.where(np.isin(label, ids), label, 0).astype(label.dtype)
    rot = ndimage.affine_transform(seg, M[:3, :3], offset=M[:3, 3], order=0,
                                   output_shape=label.shape)
    out = np.where(np.isin(label, ids), 0, label)
    moved = rot > 0
    out[moved] = rot[moved]
    return out.astype(label.dtype)


def compensate_pelvis(label, affine, *, target_pt: float = 20.0,
                      sup_axis=WORLD_SUPERIOR, lr_axis=None):
    """Phase 2.5 (voxel realisation, for the Phase-3 IMAGE) — rotate the whole bony
    spine + sacrum rigidly about the femoral-head axis (femurs = fixed ground) so PT
    falls toward `target_pt`. For the post-op ANGLES use predict_compensated_alignment
    (exact); this voxel rotation is lossy at coarse resolution and is meant for
    rendering the standing posture on real-resolution CT. No-op if PT ≤ target or the
    pelvis/femurs are unavailable."""
    from .metrics import spinopelvic_summary_from_label, femoral_head_center
    from .spine import endplate_overmask_midpoint_from_label
    label = np.asarray(label)
    s = spinopelvic_summary_from_label(label, affine)
    pt = s.get("PT")
    if pt is None or pt <= target_pt:
        return label
    L = femoral_head_center(label, affine, "femur_left", "left_hip", sup_axis=sup_axis)
    R = femoral_head_center(label, affine, "femur_right", "right_hip", sup_axis=sup_axis)
    if L is None or R is None:
        return label
    F = 0.5 * (L[0] + R[0])
    lr = unit(lr_axis) if lr_axis is not None else unit(R[0] - L[0])

    # sign: rotate the pelvic radius (M->S1 midpoint) so PT moves toward target
    m = endplate_overmask_midpoint_from_label(label, affine, "S1", sup_axis, "superior")
    th = float(np.deg2rad(pt - target_pt))
    if m is not None:
        r = np.asarray(m, float) - F
        sup_s = unit(project_out(sup_axis, lr))
        pp = angle_between(project_out(rotation_matrix(lr, th) @ r, lr), sup_s)
        pm = angle_between(project_out(rotation_matrix(lr, -th) @ r, lr), sup_s)
        if abs(pm - target_pt) < abs(pp - target_pt):
            th = -th

    present = set(int(v) for v in np.unique(label)) - {0}
    spine_ids = [LABELS[n] for n in (SPINE_CRANIOCAUDAL + ["S1", "sacrum"])
                 if LABELS[n] in present]
    return _rotate_ids(label, affine, spine_ids, F, lr, th)


def simulate_correction(label, affine, level: str, delta_deg: float, *,
                        technique: str = "alif", sup_axis=WORLD_SUPERIOR,
                        lr_axis=None, cage_id: int = CAGE_ID):
    """Return a NEW label volume with the segment at/above `level` rotated by
    `delta_deg`° of added lordosis about the technique's hinge fulcrum (pelvis fixed),
    with the hinge reconciled per technique (interbody/ACR → cage; PSO → body-wedge
    resect; SPO → mid-disc). Re-run ostk.metrics on the result for the post-op angles.
    """
    label = np.asarray(label)
    A = np.asarray(affine, dtype=float)
    present = set(int(v) for v in np.unique(label)) - {0}

    mobile = mobile_ids_for_level(level, present)
    if not mobile:
        raise ValueError(f"no mobile vertebrae present at/above {level}")
    lvl_id = LABELS[level]
    lvl_mask = label == lvl_id
    if not lvl_mask.any():
        raise ValueError(f"operative level {level} (id {lvl_id}) not in the volume")

    position, mode = TECHNIQUES.get(technique.lower(), ("posterior", "cage"))
    lr = unit(lr_axis) if lr_axis is not None else _lr_axis(label, affine, sup_axis)
    theta = _oriented_theta(label, affine, level, delta_deg, lr, sup_axis)

    F = _hinge_fulcrum(label, affine, level, position, sup_axis, lr)
    if F is None:                                    # fallback: level centroid (same angle)
        F = A[:3, :3] @ np.argwhere(lvl_mask).mean(0) + A[:3, 3]

    # rotate the mobile segment about the hinge fulcrum (rotated voxels overwrite at
    # overlaps; for PSO the anterior fulcrum makes that overlap the resected, closed
    # body wedge).
    out = _rotate_ids(label, affine, mobile, F, lr, theta)

    if mode == "cage":
        below = _vertebra_below(level)
        if below and LABELS[below] in present:
            k, plus = _si_axis_and_sign(affine, sup_axis)
            _fill_disc_cage(out, out == lvl_id, label == LABELS[below], cage_id, k, plus)

    return out.astype(label.dtype)
