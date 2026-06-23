"""ostk.surgery — synthesise a patient's spine AFTER a lordosis-restoring operation.

Phase 1: the geometric primitive. Every lordosing procedure in Greenberg Ch.73
(interbody/ALIF/LLIF, ACR, SPO, PSO) reduces to the same move — rotate the spinal
segment cranial to the operative level by Δ° in the sagittal plane and reconcile the
disc/osteotomy. This module performs that rotation on a per-vertebra LABEL volume
(the metrics are computed from the label, so re-running ostk.metrics on the output
reads back the post-op PI/SS/PT/LL — the built-in validation).

The pelvis (S1 + sacrum + femurs) is held FIXED, so PI is unchanged and LL increases
by Δ → PI−LL improves by Δ, exactly the surgical objective (Eq. 73.1). Gap/cage/bone
reconciliation and CT-intensity realism are Phase 2/3; here the bony segment moves
rigidly (mobile voxels overwrite at the closing side, the opening side becomes a gap).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometry import WORLD_SUPERIOR, rotation_matrix, unit, cobb_angle
from .labels import LABELS

# Cranial → caudal vertebral chain. The "mobile" segment for a correction at `level`
# is `level` and everything ABOVE it; S1/sacrum/femurs are never mobile (pelvic anchor).
SPINE_CRANIOCAUDAL: List[str] = (
    [f"T{n}" for n in range(1, 14)] + ["L1", "L2", "L3", "L4", "L5", "L6"]
)


def mobile_ids_for_level(level: str, present_ids) -> List[int]:
    """Label ids of the vertebrae at or cranial to `level` (the segment a correction
    at `level` swings). `level` is the lowest MOBILE vertebra — e.g. an L5–S1 ALIF is
    level='L5' (L5 and up move, S1 stays). S1/sacrum are never included."""
    if level not in SPINE_CRANIOCAUDAL:
        raise ValueError(f"level {level!r} must be one of {SPINE_CRANIOCAUDAL}")
    names = SPINE_CRANIOCAUDAL[: SPINE_CRANIOCAUDAL.index(level) + 1]
    pres = set(int(v) for v in present_ids)
    return [LABELS[n] for n in names if LABELS[n] in pres]


def _lr_axis(label, affine, sup_axis) -> np.ndarray:
    """Patient L–R (sagittal-plane normal) from the femoral-head centres; image X if
    the femurs are absent."""
    try:
        from .metrics import femoral_head_center
        L = femoral_head_center(label, affine, "femur_left", "left_hip", sup_axis=sup_axis)
        R = femoral_head_center(label, affine, "femur_right", "right_hip", sup_axis=sup_axis)
        if L is not None and R is not None:
            return unit(R[0] - L[0])
    except Exception:
        pass
    return unit(np.array([1.0, 0.0, 0.0]))


def _oriented_theta(label, affine, level, delta_deg, lr, sup_axis) -> float:
    """Signed rotation (radians) about `lr` that INCREASES lordosis by |delta_deg|.
    Chooses the sign by rotating the operative level's endplate normal and keeping the
    one that widens the level↔S1 Cobb angle (falls back to +|Δ| if S1 is unavailable)."""
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


def simulate_correction(label, affine, level: str, delta_deg: float, *,
                        sup_axis=WORLD_SUPERIOR, lr_axis=None):
    """Return a NEW label volume with the segment at/above `level` rotated by
    `delta_deg`° of added lordosis in the sagittal plane (pelvis held fixed).

    label   : per-vertebra integer label volume (the v3 scheme)
    affine  : its 4×4 voxel→world affine
    level   : lowest mobile vertebra (e.g. 'L4' for an L4–L5 interbody / L4 PSO)
    delta_deg : lordosis to add (unsigned; the extension direction is auto-chosen)

    Rotation is rigid about the operative level's centroid (the angle change is
    fulcrum-independent; the fulcrum only sets translation, which Phase 2 will use for
    gap/cage handling). Re-run ostk.metrics on the result to read the post-op angles.
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

    lr = unit(lr_axis) if lr_axis is not None else _lr_axis(label, affine, sup_axis)
    theta = _oriented_theta(label, affine, level, delta_deg, lr, sup_axis)

    # fulcrum = world centroid of the operative level
    F = A[:3, :3] @ np.argwhere(lvl_mask).mean(0) + A[:3, 3]

    # affine_transform needs the OUTPUT-index -> INPUT-index map (it pulls). World
    # forward correction is X' = R(X-F)+F (R = +theta); so output->input uses R⁻¹.
    Rinv = rotation_matrix(lr, -theta)
    Tn = np.eye(4)
    Tn[:3, :3] = Rinv
    Tn[:3, 3] = F - Rinv @ F
    M = np.linalg.inv(A) @ Tn @ A                      # index -> index

    from scipy import ndimage
    mobile_only = np.where(np.isin(label, mobile), label, 0).astype(label.dtype)
    rotated = ndimage.affine_transform(mobile_only, M[:3, :3], offset=M[:3, 3],
                                       order=0, output_shape=label.shape)
    out = np.where(np.isin(label, mobile), 0, label)   # lift the mobile segment out
    moved = rotated > 0
    out[moved] = rotated[moved]                         # set it down rotated (mobile wins overlaps)
    return out.astype(label.dtype)
