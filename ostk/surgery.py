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
    """Signed rotation (rad) that ADDS lordosis. Pick the sign that widens the actual
    LL Cobb — the L1↔S1 endplate angle (L1 = the LL top reference, in the mobile
    segment) — applying the SAME vector rotation the voxels get. Maximising the L1–S1
    Cobb (which stays <90° for any lumbar spine, so the acute value is monotonic) is
    exactly 'increase lordosis'. Falls back to +|Δ| if an endplate is unavailable."""
    th = float(np.deg2rad(abs(delta_deg)))
    present = set(int(v) for v in np.unique(label)) - {0}
    mobile = set(mobile_ids_for_level(level, present))
    ref = "L1" if LABELS["L1"] in mobile else level   # top of the mobile segment
    try:
        from .spine import endplate_from_label
        _, n_ref, _ = endplate_from_label(label, affine, ref, "superior", normal_axis=sup_axis)
        _, n_s1, _ = endplate_from_label(label, affine, "S1", "superior", normal_axis=sup_axis)
    except Exception:
        return th
    plus = cobb_angle(rotation_matrix(lr, th) @ n_ref, n_s1, lr)
    minus = cobb_angle(rotation_matrix(lr, -th) @ n_ref, n_s1, lr)
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


def correction_transform(label, affine, level: str, delta_deg: float, *,
                         technique: str = "alif", sup_axis=WORLD_SUPERIOR, lr_axis=None,
                         flip: bool = False):
    """Resolve the correction into its geometric pieces (shared by the label and CT
    paths): the mobile vertebra ids, the hinge fulcrum F (world), the L–R rotation
    axis, the signed angle θ, and the technique's (fulcrum position, reconcile mode)."""
    label = np.asarray(label)
    A = np.asarray(affine, dtype=float)
    present = set(int(v) for v in np.unique(label)) - {0}
    mobile = mobile_ids_for_level(level, present)
    if not mobile:
        raise ValueError(f"no mobile vertebrae present at/above {level}")
    lvl_mask = label == LABELS[level]
    if not lvl_mask.any():
        raise ValueError(f"operative level {level} not in the volume")
    position, mode = TECHNIQUES.get(technique.lower(), ("posterior", "cage"))
    lr = unit(lr_axis) if lr_axis is not None else _lr_axis(label, affine, sup_axis)
    theta = _oriented_theta(label, affine, level, delta_deg, lr, sup_axis)
    if flip:                                          # caller-forced opposite direction
        theta = -theta
    F = _hinge_fulcrum(label, affine, level, position, sup_axis, lr)
    if F is None:
        F = A[:3, :3] @ np.argwhere(lvl_mask).mean(0) + A[:3, 3]
    return {"mobile_ids": mobile, "F": F, "lr": lr, "theta": theta,
            "position": position, "mode": mode}


def warp_ct(ct, label, affine, level: str, delta_deg: float, *,
            technique: str = "alif", sup_axis=WORLD_SUPERIOR, lr_axis=None,
            postop_label=None, cage_hu: float = 250.0, cage_id: int = CAGE_ID,
            flip: bool = False):
    """Phase 3 — synthesise the post-op CT IMAGE. The mobile vertebral segment moves
    rigidly with its labels; surrounding soft tissue deforms SMOOTHLY so there is no
    seam, via a displacement field weighted w∈[0,1] (1 at mobile bone → full rigid
    motion, 0 at fixed bone → no motion, a distance-blended transition between). If
    `postop_label` is given, its cage voxels are stamped at `cage_hu` (instrumentation).
    Returns the warped CT (same shape/dtype family). Heavy on full-res volumes — meant
    to be precomputed offline (e.g. for the demo), not run live."""
    from scipy import ndimage
    ct = np.asarray(ct)
    label = np.asarray(label)
    A = np.asarray(affine, dtype=float)
    t = correction_transform(label, affine, level, delta_deg, technique=technique,
                             sup_axis=sup_axis, lr_axis=lr_axis, flip=flip)
    F, lr, theta = t["F"], t["lr"], t["theta"]
    Rinv = rotation_matrix(lr, -theta)

    mobile_mask = np.isin(label, t["mobile_ids"])
    fixed_bone = (label > 0) & ~mobile_mask
    spacing = np.abs(np.linalg.norm(A[:3, :3], axis=0))
    d_mob = ndimage.distance_transform_edt(~mobile_mask, sampling=spacing)
    d_fix = ndimage.distance_transform_edt(~fixed_bone, sampling=spacing) if fixed_bone.any() \
        else np.full(label.shape, 1e3)
    w = (d_fix / (d_fix + d_mob + 1e-6)).astype(np.float32)     # 1@mobile, 0@fixed

    # PULL warp: out[y] = ct[ y + w·(R⁻¹(y−F)+F − y) ] (identity where w=0, rigid where w=1)
    sh = ct.shape
    grid = np.stack(np.meshgrid(np.arange(sh[0]), np.arange(sh[1]),
                                np.arange(sh[2]), indexing="ij"), -1).astype(np.float32)
    world = grid @ A[:3, :3].T.astype(np.float32) + A[:3, 3].astype(np.float32)
    rot_inv = (world - F) @ Rinv.T.astype(np.float32) + F.astype(np.float32)
    src_world = world + w[..., None] * (rot_inv - world)
    invA = np.linalg.inv(A)
    src_vox = src_world @ invA[:3, :3].T.astype(np.float32) + invA[:3, 3].astype(np.float32)
    warped = ndimage.map_coordinates(
        ct, [src_vox[..., 0], src_vox[..., 1], src_vox[..., 2]],
        order=1, mode="constant", cval=float(ct.min()))

    if postop_label is not None:                               # stamp instrumentation
        warped = warped.copy()
        warped[np.asarray(postop_label) == cage_id] = cage_hu
    return warped.astype(ct.dtype)


def bend_spine(volume, affine, total_delta_deg, *, label_for_axes=None,
               sup_axis=WORLD_SUPERIOR, lr_axis=None, order=1, cval=None,
               z_lo_name="S1", z_hi_name="L1", out_affine=None, out_shape=None):
    """Smooth, DISTRIBUTED lordosis correction (the natural-looking synthesis).

    Instead of pivoting the whole upper spine about one disc (a sharp kink), the
    extension angle RAMPS continuously from 0 at the sacrum to `total_delta_deg` at L1
    (smoothstep over the lumbar span) — so every lumbar level shares the correction and
    the result is a smooth lordotic curve with no gaps to fill. One per-voxel rotation
    field about the L–R axis (Rodrigues), pull-resampled in a single pass; apply the
    SAME call to the label (order=0) and the CT (order=1) so they stay registered.

    `label_for_axes` supplies the segmentation when `volume` is the CT. Pelvis (below S1)
    is untouched, so PI is invariant; the thoracic above L1 is carried rigidly.

    `out_affine`/`out_shape`: warp a FULL-RES `volume` directly onto a different (e.g.
    downsampled demo) output grid in one resample — memory-light (the grid is the small
    output), so the post-op is generated from full-res but ships at demo resolution."""
    from scipy import ndimage
    vol = np.asarray(volume)
    out_aff = np.asarray(out_affine, float) if out_affine is not None else None
    out_sh = tuple(out_shape) if out_shape is not None else None
    lab = np.asarray(label_for_axes if label_for_axes is not None else volume)
    A = np.asarray(affine, float)
    lr = unit(lr_axis) if lr_axis is not None else _lr_axis(lab, affine, sup_axis)
    sup_s = unit(project_out(sup_axis, lr))

    def _world(ids):
        m = np.isin(lab, ids)
        if not m.any():
            return None
        idx = np.argwhere(m)
        return idx @ A[:3, :3].T + A[:3, 3]
    s1w = _world([LABELS[z_lo_name]])
    l1w = _world([LABELS[z_hi_name]])
    lumw = _world([LABELS[n] for n in ("L1", "L2", "L3", "L4", "L5", "L6", z_lo_name)
                   if n in LABELS])
    if s1w is None or l1w is None or lumw is None:
        return vol                                    # need the span anchors
    z_lo = float((s1w @ sup_s).max())                 # top of S1 — bend starts here
    z_hi = float((l1w @ sup_s).max())                 # top of L1 — full correction
    if z_hi - z_lo < 1e-3:
        return vol
    # fulcrum: posterior column at the lumbosacral base (so the anterior column opens)
    ant = unit(project_out(np.array([0.0, 1.0, 0.0]), lr))
    if ant[1] < 0:
        ant = -ant
    c = lumw.mean(0)
    F = (c - ((c - s1w.mean(0)) @ sup_s) * sup_s          # drop to S1 height
         - (float((lumw @ ant).max()) - float(c @ ant)) * ant)   # back to posterior edge

    # sign that ADDS lordosis (widen L1↔S1 Cobb), same robust test as _oriented_theta
    sgn = 1.0
    try:
        from .spine import endplate_from_label
        _, n1, _ = endplate_from_label(lab, affine, "L1", "superior", normal_axis=sup_axis)
        _, n7, _ = endplate_from_label(lab, affine, z_lo_name, "superior", normal_axis=sup_axis)
        if cobb_angle(rotation_matrix(lr, 0.05) @ n1, n7, lr) < \
           cobb_angle(rotation_matrix(lr, -0.05) @ n1, n7, lr):
            sgn = -1.0
    except Exception:
        pass
    total = sgn * np.deg2rad(total_delta_deg)

    # output grid (default = input grid; or a separate demo grid for full-res→demo warp)
    gsh = out_sh if out_sh is not None else vol.shape
    gaff = out_aff if out_aff is not None else A
    grid = np.stack(np.meshgrid(np.arange(gsh[0]), np.arange(gsh[1]),
                                np.arange(gsh[2]), indexing="ij"), -1).astype(np.float32)
    world = grid @ gaff[:3, :3].T.astype(np.float32) + gaff[:3, 3].astype(np.float32)
    h = world @ sup_s.astype(np.float32)
    t = np.clip((h - z_lo) / (z_hi - z_lo), 0.0, 1.0)
    theta = (total * (t * t * (3.0 - 2.0 * t))).astype(np.float32)   # smoothstep ramp
    a = -theta                                        # PULL: output->input is the inverse
    ca, sa = np.cos(a), np.sin(a)
    d = (world - F.astype(np.float32))
    lrf = lr.astype(np.float32)
    dxl = np.cross(np.broadcast_to(lrf, d.shape), d)
    ddl = (d @ lrf)[..., None]
    d_rot = d * ca[..., None] + dxl * sa[..., None] + lrf * ddl * (1.0 - ca)[..., None]
    src_world = F.astype(np.float32) + d_rot
    invA = np.linalg.inv(A)
    src_vox = src_world @ invA[:3, :3].T.astype(np.float32) + invA[:3, 3].astype(np.float32)
    if cval is None:
        cval = float(vol.min())
    out = ndimage.map_coordinates(vol, [src_vox[..., 0], src_vox[..., 1], src_vox[..., 2]],
                                  order=order, mode="constant", cval=cval)
    return out.astype(vol.dtype)


def age_adjusted_targets(pi: float, age: float) -> dict:
    """Lafage age-adjusted ideal sagittal alignment (Lafage et al., Spine 2016) — the
    'ideal' loosens with age, so a correction won't over-flatten an older spine (a known
    driver of proximal junctional kyphosis). PI−LL=(age−55)/2+3, PT=(age−55)/3+20,
    SVA=2(age−55)+25 mm; target LL = PI − (age-adjusted PI−LL)."""
    pill = (age - 55.0) / 2.0 + 3.0
    pt = (age - 55.0) / 3.0 + 20.0
    sva = 2.0 * (age - 55.0) + 25.0
    return {"PI-LL": round(pill, 1), "PT": round(pt, 1), "SVA_mm": round(sva, 1),
            "LL": round(pi - pill, 1)}


def plan_realignment(summary: dict, age: float, *, reciprocal_k: float = 0.5) -> dict:
    """Turn pre-op PI/LL/PT + age into a biomechanically-grounded correction:
      • ΔLL to reach the age-adjusted LL target (Lafage);
      • reciprocal thoracic ΔTK ≈ k·ΔLL (literature 0.34–0.58·ΔPI−LL; flexible ~0.58);
      • pelvic anteversion to release retroversion toward the age-adjusted PT.
    Returns the deltas + the age-adjusted target dict."""
    pi, ll, pt = summary.get("PI"), summary.get("LL"), summary.get("PT")
    tgt = age_adjusted_targets(pi, age)
    d_ll = max(0.0, round(tgt["LL"] - ll, 2)) if (pi is not None and ll is not None) else 0.0
    d_tk = round(reciprocal_k * d_ll, 2)
    antevert = max(0.0, round(pt - tgt["PT"], 2)) if pt is not None else 0.0
    return {"delta_ll": d_ll, "delta_tk": d_tk, "pelvic_antevert": antevert,
            "reciprocal_k": reciprocal_k, "age": age, "targets": tgt}


def _rodrigues(d, lr, angle):
    """Rotate vectors d (…,3) about unit axis lr by `angle` (scalar OR per-voxel array)."""
    ca, sa = np.cos(angle), np.sin(angle)
    if np.ndim(ca) == 0:
        ca_, sa_, cc = float(ca), float(sa), 1.0 - float(ca)
    else:
        ca_, sa_, cc = ca[..., None], sa[..., None], (1.0 - ca)[..., None]
    dxl = np.cross(np.broadcast_to(lr, d.shape), d)
    ddl = (d @ lr)[..., None]
    return d * ca_ + dxl * sa_ + lr * ddl * cc


def place_interbody_cages(label, ct, affine, disc_pairs, *, cage_id=CAGE_ID,
                          cage_hu=350.0, gap_iters=3):
    """Render an interbody CAGE in each operated disc — the surgical hardware of an
    ALIF/LLIF/TLIF/ACR (anterior/lateral interbody fusion); NO vertebral body is
    resected (that is PSO only). The disc space is the thin gap between two adjacent
    vertebra masks (their dilated overlap); we fill it with a cage label + a bright-ish
    implant HU so the post-op CT shows a device at every fused level. Returns
    (label, ct) copies. `disc_pairs` = [(upper, lower), …] cranial→caudal."""
    from scipy import ndimage
    lab = np.asarray(label).copy()
    im = np.asarray(ct).copy()
    for up, lo in disc_pairs:
        if up not in LABELS or lo not in LABELS:
            continue
        um, lm = lab == LABELS[up], lab == LABELS[lo]
        if not um.any() or not lm.any():
            continue
        disc = (ndimage.binary_dilation(um, iterations=gap_iters)
                & ndimage.binary_dilation(lm, iterations=gap_iters) & (lab == 0))
        lab[disc] = cage_id
        im[disc] = cage_hu
    return lab, im


def bend_params(label, affine, *, delta_ll, delta_tk=0.0, pelvic_antevert=0.0,
                sup_axis=WORLD_SUPERIOR, lr_axis=None):
    """Compute the WORLD-SPACE parameters of the post-op bend ONCE (L–R axis, lumbosacral
    fulcrum, height anchors, signed angle ramp, pelvic hinge). They are in mm and grid-
    independent, so derive them on a fast COARSE segmentation and reuse them for both the
    full-res image warp (`synthesize_postop`) and the analytic overlay/number carry-forward
    (`transform_world_points`). Returns a dict, or None if the span anchors are missing."""
    lab = np.asarray(label)
    A = np.asarray(affine, float)
    lr = unit(lr_axis) if lr_axis is not None else _lr_axis(lab, affine, sup_axis)
    sup_s = unit(project_out(sup_axis, lr))

    def _w(ids):
        m = np.isin(lab, ids)
        return (np.argwhere(m) @ A[:3, :3].T + A[:3, 3]) if m.any() else None
    s1w, l1w = _w([LABELS["S1"]]), _w([LABELS["L1"]])
    lumw = _w([LABELS[n] for n in ("L1", "L2", "L3", "L4", "L5", "L6", "S1") if n in LABELS])
    tw = _w([LABELS[f"T{n}"] for n in range(1, 14) if f"T{n}" in LABELS])
    if s1w is None or l1w is None or lumw is None:
        return None
    z_s1 = float((s1w @ sup_s).max())
    z_l1 = float((l1w @ sup_s).max())
    z_tt = float((tw @ sup_s).max()) if tw is not None else z_l1 + (z_l1 - z_s1)

    ant = unit(project_out(np.array([0.0, 1.0, 0.0]), lr))
    if ant[1] < 0:
        ant = -ant
    c = lumw.mean(0)
    F_lum = (c - ((c - s1w.mean(0)) @ sup_s) * sup_s
             - (float((lumw @ ant).max()) - float(c @ ant)) * ant)

    # extension sign (adds lordosis); thoracic flexion is the opposite cumulative sense
    sgn = 1.0
    try:
        from .spine import endplate_from_label
        _, n1, _ = endplate_from_label(lab, affine, "L1", "superior", normal_axis=sup_axis)
        _, n7, _ = endplate_from_label(lab, affine, "S1", "superior", normal_axis=sup_axis)
        if cobb_angle(rotation_matrix(lr, 0.05) @ n1, n7, lr) < \
           cobb_angle(rotation_matrix(lr, -0.05) @ n1, n7, lr):
            sgn = -1.0
    except Exception:
        pass
    zs = np.array([z_s1, z_l1, max(z_tt, z_l1 + 1.0)], dtype=float)
    ang = sgn * np.deg2rad(np.array([0.0, delta_ll, delta_ll - delta_tk], dtype=float))

    # global pelvic anteversion about the femoral-head axis (sign reduces PT)
    F_hip, a_pelvis = None, 0.0
    if pelvic_antevert:
        try:
            from .metrics import femoral_head_center
            from .spine import endplate_overmask_midpoint_from_label
            L = femoral_head_center(lab, affine, "femur_left", "left_hip", sup_axis=sup_axis)
            R = femoral_head_center(lab, affine, "femur_right", "right_hip", sup_axis=sup_axis)
            m = endplate_overmask_midpoint_from_label(lab, affine, "S1", sup_axis, "superior")
            if L is not None and R is not None and m is not None:
                F_hip = 0.5 * (L[0] + R[0])
                r = np.asarray(m, float) - F_hip
                th = np.deg2rad(pelvic_antevert)
                pp = angle_between(project_out(rotation_matrix(lr, th) @ r, lr), sup_s)
                pm = angle_between(project_out(rotation_matrix(lr, -th) @ r, lr), sup_s)
                a_pelvis = -th if pm < pp else th       # whichever lowers PT
        except Exception:
            F_hip = None
    return {"lr": lr, "sup_s": sup_s, "F_lum": F_lum, "zs": zs, "ang": ang,
            "F_hip": F_hip, "a_pelvis": a_pelvis, "z_s1": z_s1, "z_l1": z_l1,
            "z_tt": z_tt, "sgn": float(sgn)}


def transform_world_points(points, params):
    """FORWARD post-op transform of world-mm points (pre-op → post-op): bend each point by
    +θ about the lumbosacral fulcrum (θ from its source height) then antevert by the pelvic
    hinge. This carries the CLEAN pre-op construction overlay onto the post-op spine without
    re-fitting the resampled (slightly rough) image. Accepts a single point or an (N,3)."""
    p = np.asarray(points, float)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    lr, sup_s, F_lum = params["lr"], params["sup_s"], params["F_lum"]
    theta = np.interp(p @ sup_s, params["zs"], params["ang"])
    q = F_lum + _rodrigues(p - F_lum, lr, theta)
    if params.get("F_hip") is not None and params.get("a_pelvis"):
        q = params["F_hip"] + _rodrigues(q - params["F_hip"], lr, params["a_pelvis"])
    return q[0] if single else q


def synthesize_postop(volume, affine, *, params=None, delta_ll=None, delta_tk=0.0,
                      pelvic_antevert=0.0, label_for_axes=None, sup_axis=WORLD_SUPERIOR,
                      lr_axis=None, order=1, cval=None, out_affine=None, out_shape=None):
    """Biomechanically-grounded post-op IMAGE synthesis (Phase 4). ONE composite field:
    lumbar lordosis (0→ΔLL over S1→L1) + reciprocal thoracic kyphosis (ΔLL→ΔLL−ΔTK over
    L1→top-thoracic) + optional global pelvic anteversion. Same field warps the label
    (order 0) and CT (order 1); `out_affine`/`out_shape` warp a full-res input onto a
    (downsampled) demo grid. Pass `params` from `bend_params` (computed on a coarse seg) to
    skip the slow full-res axis derivation. NOTE: re-measuring the resampled output is
    unreliable (rough endplates) — carry numbers/overlay via `transform_world_points`."""
    from scipy import ndimage
    vol = np.asarray(volume)
    A = np.asarray(affine, float)
    if params is None:
        params = bend_params(label_for_axes if label_for_axes is not None else volume, affine,
                             delta_ll=delta_ll, delta_tk=delta_tk,
                             pelvic_antevert=pelvic_antevert, sup_axis=sup_axis, lr_axis=lr_axis)
    if params is None:
        return vol
    lr = params["lr"].astype(np.float32)
    sup_s = params["sup_s"].astype(np.float32)
    F_lum = params["F_lum"].astype(np.float32)
    zs, ang = params["zs"], params["ang"]
    F_hip, a_pelvis = params["F_hip"], params["a_pelvis"]

    gsh = tuple(out_shape) if out_shape is not None else vol.shape
    gaff = np.asarray(out_affine, float) if out_affine is not None else A
    grid = np.stack(np.meshgrid(np.arange(gsh[0]), np.arange(gsh[1]),
                                np.arange(gsh[2]), indexing="ij"), -1).astype(np.float32)
    world = grid @ gaff[:3, :3].T.astype(np.float32) + gaff[:3, 3].astype(np.float32)
    Y1 = world
    if F_hip is not None and a_pelvis:                  # undo global pelvic rotation
        Y1 = F_hip.astype(np.float32) + _rodrigues(world - F_hip.astype(np.float32), lr, -a_pelvis)
    h = Y1 @ sup_s                                       # undo the bend (per-height angle)
    theta = np.interp(h, zs, ang).astype(np.float32)
    src = F_lum + _rodrigues(Y1 - F_lum, lr, -theta)
    invA = np.linalg.inv(A)
    sv = src @ invA[:3, :3].T.astype(np.float32) + invA[:3, 3].astype(np.float32)
    if cval is None:
        cval = float(vol.min())
    out = ndimage.map_coordinates(vol, [sv[..., 0], sv[..., 1], sv[..., 2]],
                                  order=order, mode="constant", cval=cval)
    return out.astype(vol.dtype)


def simulate_correction(label, affine, level: str, delta_deg: float, *,
                        technique: str = "alif", sup_axis=WORLD_SUPERIOR,
                        lr_axis=None, cage_id: int = CAGE_ID, flip: bool = False):
    """Return a NEW label volume with the segment at/above `level` rotated by
    `delta_deg`° of added lordosis about the technique's hinge fulcrum (pelvis fixed),
    with the hinge reconciled per technique (interbody/ACR → cage; PSO → body-wedge
    resect; SPO → mid-disc). Re-run ostk.metrics on the result for the post-op angles.
    """
    label = np.asarray(label)
    present = set(int(v) for v in np.unique(label)) - {0}
    t = correction_transform(label, affine, level, delta_deg, technique=technique,
                             sup_axis=sup_axis, lr_axis=lr_axis, flip=flip)

    # rotate the mobile segment about the hinge fulcrum (rotated voxels overwrite at
    # overlaps; for PSO the anterior fulcrum makes that overlap the resected, closed
    # body wedge).
    out = _rotate_ids(label, affine, t["mobile_ids"], t["F"], t["lr"], t["theta"])

    if t["mode"] == "cage":
        below = _vertebra_below(level)
        if below and LABELS[below] in present:
            k, plus = _si_axis_and_sign(affine, sup_axis)
            _fill_disc_cage(out, out == LABELS[level], label == LABELS[below], cage_id, k, plus)

    return out.astype(label.dtype)
