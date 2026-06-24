"""ostk.metrics — measurements composed from the primitives.

Flagship: pelvic incidence (PI) + sacral slope (SS) + pelvic tilt (PT). PI is the
one sagittal parameter valid on supine CT (posture-invariant). Convention:
endplate normal oriented cranially; pelvic radius = hip-axis → sacral-endplate
midpoint; all angles taken in the patient sagittal plane (normal ⟂ the L–R
bicoxofemoral axis). The absolute convention is to be confirmed against manual
radiographic PI (Paper 2, Aim 2) — the geometry/identity is unit-tested.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .geometry import (WORLD_SUPERIOR, angle_between, cobb_angle, fit_plane_tls,
                       fit_sphere, project_out, signed_angle_in_plane, unit)
from .spine import fit_endplate
from .record import Measurement

PI_METHOD_VERSION = "pi-v1"
LL_METHOD_VERSION = "ll-v1"

# Superior endplates, cranial → caudal, that bound lumbar lordosis (Greenberg
# Fig. 73.1: L1 superior endplate → S1 superior endplate). Per-segment lordosis
# is taken between consecutive entries.
LL_ENDPLATE_CHAIN = ("L1", "L2", "L3", "L4", "L5", "S1")


def pelvic_incidence(endplate_points, femhead_left_points, femhead_right_points,
                     sup_axis=WORLD_SUPERIOR) -> Dict:
    """PI/SS/PT from three world-mm point clouds: the S1 superior endplate and
    the two femoral heads. Returns a dict with the angles, landmarks, residuals."""
    m, n, ep_rms = fit_plane_tls(endplate_points)
    cL, rL, eL = fit_sphere(femhead_left_points)       # caller supplies head clouds
    cR, rR, eR = fit_sphere(femhead_right_points)
    return _pi_from_plane(m, n, ep_rms, cL, cR, sup_axis, rL=rL, rR=rR, eL=eL, eR=eR)


def femoral_head_center(label, affine, femur_name, hip_name=None, *,
                        sup_axis=WORLD_SUPERIOR, contact_mm=6.0,
                        slab_frac=0.30, min_voxels=50):
    """Robust femoral-head CENTRE (the hip-axis endpoint for PI/PT).

    The femoral head is a sphere, but the femur mask also holds the neck and
    proximal shaft, which drag a naive sphere fit off the head (centre pulled
    toward the neck). Two-stage fit:
      1. SEED from the acetabular interface — femur voxels within `contact_mm` of
         the same-side hip mask are, by construction, on the head surface (they sit
         in the socket), so a sphere through them is already well-centred.
      2. EXTEND through the neck — keep femur voxels on that sphere's SHELL
         (|‖p−c‖−r| small, within ~1.3 r) and refit ×2. This grows the fit over the
         whole head while rejecting the neck/shaft — the same off-surface rejection
         the endplate fit uses for osteophytes.
    Falls back to a robust cranial-slab fit when the hip mask is absent/too small.
    Returns (centre, radius, rms) or None."""
    from .labels import lid, LABELS
    from .masks import binary_mask, largest_component, mask_world, surface_slab
    fem = mask_world(largest_component(binary_mask(label, lid(femur_name))), affine)
    if len(fem) < min_voxels:
        return None

    def _refine(c, r):
        shell = None
        for _ in range(2):
            near = fem[np.linalg.norm(fem - c, axis=1) <= 1.3 * r]
            if len(near) < min_voxels:
                break
            d = np.linalg.norm(near - c, axis=1)
            s = near[np.abs(d - r) <= max(0.18 * r, 3.0)]
            if len(s) < min_voxels:
                break
            shell = s
            c, r, _ = fit_sphere(shell)
        # rms over the HEAD shell (fit quality), not the whole femur (shaft is far)
        pts = shell if shell is not None else fem
        rms = float(np.sqrt(np.mean((np.linalg.norm(pts - c, axis=1) - r) ** 2)))
        return c, r, rms

    seed = None
    if hip_name and hip_name in LABELS:
        hipm = binary_mask(label, lid(hip_name))
        if hipm.any():
            hip = mask_world(hipm, affine)
            try:
                from scipy.spatial import cKDTree
                d, _ = cKDTree(hip).query(fem)
                cand = fem[d <= contact_mm]
                if len(cand) >= min_voxels:
                    seed = cand                        # acetabular-contact head surface
            except Exception:
                seed = None
    if seed is None:                                   # fallback: cranial slab
        seed = surface_slab(fem, sup_axis, "superior", slab_frac)
        if len(seed) < min_voxels:
            return None
    c, r, _ = fit_sphere(seed)
    return _refine(c, r)


def _pi_from_plane(m, n, ep_rms, cL, cR, sup_axis=WORLD_SUPERIOR,
                   rL=None, rR=None, eL=None, eR=None) -> Dict:
    """PI/SS/PT from a PRECOMPUTED S1 endplate plane (centroid m, normal n, rms)
    plus the two femoral-head CENTRES (cL, cR; from `femoral_head_center`). Sharing
    this lets the PI core and the LL path use the SAME endplate primitive, so the
    sacral slope is consistent."""
    cL = np.asarray(cL, float); cR = np.asarray(cR, float)
    bicox = 0.5 * (cL + cR)

    lr = unit(cR - cL)                                  # left–right axis
    n_s = unit(project_out(n, lr))                     # endplate normal in sagittal plane
    radius = project_out(m - bicox, lr)                # hip axis → endplate midpoint
    sup_s = unit(project_out(sup_axis, lr))            # vertical in sagittal plane
    if n_s @ sup_s < 0:                                # orient endplate normal cranially
        n_s = -n_s

    PI = angle_between(n_s, radius)
    SS = angle_between(n_s, sup_s)
    PT = angle_between(radius, sup_s)
    return {
        "PI": PI, "SS": SS, "PT": PT,
        "landmarks_world_mm": {
            "endplate_midpoint": m.tolist(),
            "endplate_normal": n_s.tolist(),
            "femhead_left": cL.tolist(), "femhead_right": cR.tolist(),
            "bicoxofemoral": bicox.tolist()},
        "fit_residuals": {
            "s1_endplate_rms": ep_rms,
            "femhead_left_rms": eL, "femhead_right_rms": eR,
            "femhead_left_radius": rL, "femhead_right_radius": rR},
    }


def _pi_from_label_core(label, affine, sup_axis, endplate_frac, head_frac,
                        min_voxels):
    """Extract the PI/SS/PT result dict from a v3 label volume (shared by the
    PI Measurement and the spinopelvic summary). The S1 superior endplate uses the
    shared `ostk.spine` endplate primitive (anterior band + true top-surface fit —
    the SAME one as lumbar lordosis, so the sacral slope is consistent and reads
    the true tilt instead of under-reading it with a flat slab). Femoral-head
    centres use the robust acetabular-interface sphere fit (`femoral_head_center`),
    not a cranial slab. `endplate_frac` is kept for signature compatibility but no
    longer used. Returns (result_dict_or_None, flags)."""
    from .spine import endplate_from_label, endplate_overmask_midpoint_from_label

    flags: list = []
    ep_plane = endplate_from_label(label, affine, "S1", "superior",
                                   normal_axis=sup_axis, min_points=min_voxels)
    L = femoral_head_center(label, affine, "femur_left", "left_hip",
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)
    R = femoral_head_center(label, affine, "femur_right", "right_hip",
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)

    if ep_plane is None:
        flags.append("low_voxels:S1")
    if L is None:
        flags.append("low_voxels:femur_left")
    if R is None:
        flags.append("low_voxels:femur_right")
    if flags:
        return None, flags

    (cL, rL, eL), (cR, rR, eR) = L, R
    m, n, ep_rms = ep_plane
    # PI/PT radius origin = midpoint of the endplate portion over the body, on the rim
    # (more accurate than the corner-midpoint, which the anterior tangent skip biases
    # posterior). Orientation (n) is unchanged, so SS/LL are unaffected.
    om = endplate_overmask_midpoint_from_label(label, affine, "S1", sup_axis, "superior")
    if om is not None:
        m = om
    r = _pi_from_plane(m, n, ep_rms, cL, cR, sup_axis, rL=rL, rR=rR, eL=eL, eR=eR)
    if abs(r["SS"] + r["PT"] - r["PI"]) > 1.0:         # geometric identity check
        flags.append("identity_violation")
    return r, (flags or ["ok"])


def pelvic_incidence_from_label(label, affine, *, case_id: str = "",
                                sup_axis=WORLD_SUPERIOR, endplate_frac: float = 0.15,
                                head_frac: float = 0.35,
                                min_voxels: int = 50) -> Measurement:
    """Compose PI from a v3 label volume. Returns a Measurement with QC flags
    (never silently drops a bad case). SS/PT are available via
    `spinopelvic_summary_from_label`."""
    r, flags = _pi_from_label_core(label, affine, sup_axis, endplate_frac,
                                   head_frac, min_voxels)
    if r is None:
        return Measurement(case_id=case_id, parameter="pelvic_incidence",
                           value=None, qc_flags=flags,
                           method_version=PI_METHOD_VERSION)
    return Measurement(
        case_id=case_id, parameter="pelvic_incidence", value=round(r["PI"], 3),
        landmarks_world_mm=r["landmarks_world_mm"], fit_residuals=r["fit_residuals"],
        qc_flags=flags, method_version=PI_METHOD_VERSION)


# ---------------------------------------------------------------------------
# Lumbar lordosis (LL) — Greenberg §73.5.3, Fig. 73.1. Supine surrogate: the
# Cobb construction is exact, but the absolute value differs from a standing
# film, so every LL Measurement carries supine_ct=True (SPEC §5).
# ---------------------------------------------------------------------------

def lumbar_lordosis(endplate_normals: Dict[str, np.ndarray], lr_axis) -> Dict:
    """LL from a dict of *cranially-oriented* endplate-plane normals (keys are a
    cranial→caudal chain such as ``LL_ENDPLATE_CHAIN``) and the patient L–R axis
    (the sagittal-plane normal — e.g. the bicoxofemoral vector).

    Returns total LL (Cobb magnitude between the first and last endplate present)
    plus signed per-segment lordosis between consecutive present endplates
    (positive = lordotic about the right-hand rule on `lr_axis`)."""
    lr = unit(lr_axis)
    present = [lv for lv in LL_ENDPLATE_CHAIN if lv in endplate_normals]
    if len(present) < 2:
        return {"LL": None, "segments": {}, "levels": present}
    top, bot = present[0], present[-1]
    LL = cobb_angle(endplate_normals[top], endplate_normals[bot], lr)
    segments = {}
    for a, b in zip(present[:-1], present[1:]):
        segments[f"{a}-{b}"] = round(
            signed_angle_in_plane(endplate_normals[a], endplate_normals[b], lr), 3)
    return {"LL": round(LL, 3), "segments": segments, "levels": present,
            "span": f"{top}-{bot}"}


def _endplate_normal_from_label(label, affine, level, which, sup_axis, frac,
                                min_voxels):
    """(unit normal oriented cranially, centroid, rms, n_points) for one vertebral
    endplate, or (None, …) if the level is absent / too small. Delegates to the
    `ostk.spine.fit_endplate` primitive (anterior-body + true-surface fit)."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world
    from .spine import corner_params_for_level
    allpts = mask_world(largest_component(binary_mask(label, lid(level))), affine)
    if len(allpts) < min_voxels:
        return None, None, None, len(allpts)
    res = fit_endplate(allpts, sup_axis, which, min_points=min_voxels,
                       **corner_params_for_level(level))
    if res is None:
        return None, None, None, len(allpts)
    c, n, rms = res
    return n, c, rms, len(allpts)


def _lr_axis_from_label(label, affine, sup_axis, head_frac, min_voxels):
    """Patient L–R (sagittal-plane normal) from the two femoral-head centres
    (bicoxofemoral vector, robust acetabular-interface fit). Returns (lr_unit, ok).
    Falls back to the image X axis with ok=False if a femur is missing."""
    L = femoral_head_center(label, affine, "femur_left", "left_hip",
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)
    R = femoral_head_center(label, affine, "femur_right", "right_hip",
                            sup_axis=sup_axis, slab_frac=head_frac, min_voxels=min_voxels)
    if L is None or R is None:
        return unit(np.array([1.0, 0.0, 0.0])), False
    return unit(R[0] - L[0]), True                      # right − left


def lumbar_lordosis_from_label(label, affine, *, case_id: str = "",
                               sup_axis=WORLD_SUPERIOR, endplate_frac: float = 0.15,
                               head_frac: float = 0.35, min_voxels: int = 30
                               ) -> Measurement:
    """Compose lumbar lordosis from a v3 label volume. Sagittal plane is derived
    from the femoral heads (data-derived L–R axis, robust to scan tilt; SPEC §3);
    each endplate normal is a TLS fit to that body's cranial slab. Needs at least
    L1 + S1; missing intermediate levels are skipped (and flagged) so a
    FOV-clipped scan still yields the L1–S1 Cobb where possible."""
    flags: list = []
    lr, ok = _lr_axis_from_label(label, affine, sup_axis, head_frac, min_voxels)
    if not ok:
        flags.append("sagittal_ref_fallback")          # used image X, not femurs

    normals: Dict[str, np.ndarray] = {}
    residuals: Dict[str, float] = {}
    landmarks: Dict[str, list] = {}
    for lv in LL_ENDPLATE_CHAIN:
        n, c, rms, k = _endplate_normal_from_label(
            label, affine, lv, "superior", sup_axis, endplate_frac, min_voxels)
        if n is None:
            flags.append(f"missing_label:{lv}")
            continue
        normals[lv] = n
        residuals[f"{lv}_endplate_rms"] = round(rms, 3)
        landmarks[f"{lv}_superior_endplate"] = c.tolist()

    if "L1" not in normals or "S1" not in normals:
        flags.append("LL_span_unavailable")
        return Measurement(case_id=case_id, parameter="lumbar_lordosis",
                           value=None, qc_flags=flags or ["ok"],
                           method_version=LL_METHOD_VERSION)

    r = lumbar_lordosis(normals, lr)
    landmarks["per_segment_lordosis_deg"] = r["segments"]
    flags = flags or ["ok"]
    return Measurement(
        case_id=case_id, parameter="lumbar_lordosis", value=r["LL"],
        landmarks_world_mm=landmarks, fit_residuals=residuals,
        qc_flags=flags, method_version=LL_METHOD_VERSION, supine_ct=True)


# ---------------------------------------------------------------------------
# Alignment targets & SRS-Schwab sagittal modifiers (Greenberg §73.6 / §73.7.2).
# Pure scalar functions — no geometry — so they are exhaustively testable against
# the published thresholds. SVA is out of scope on supine, C7-less CT (SPEC §5);
# pass sva_cm only if it comes from elsewhere.
# ---------------------------------------------------------------------------

def _schwab_grade(value, lo: float, hi: float) -> str:
    """SRS-Schwab 3-level modifier: 0 (<lo), + (lo..hi), ++ (>hi)."""
    if value < lo:
        return "0"
    return "+" if value <= hi else "++"


def pi_ll_mismatch(pi: float, ll: float) -> Dict:
    """PI − LL mismatch (Greenberg: the dominant driver of sagittal imbalance).
    Objective is LL = PI ± 9°; surgical-target flag at |PI−LL| > 9°. Schwab
    PI–LL modifier: 0 (<10°), + (10–20°), ++ (>20°)."""
    mm = pi - ll
    return {
        "pi_minus_ll": round(mm, 3),
        "abs_pi_minus_ll": round(abs(mm), 3),
        "ll_target_deg": [round(pi - 9.0, 1), round(pi + 9.0, 1)],   # LL = PI ± 9°
        "within_target_9deg": abs(mm) <= 9.0,          # Greenberg LL = PI ± 9°
        "surgical_target": abs(mm) > 9.0,
        "schwab_modifier": _schwab_grade(abs(mm), 10.0, 20.0),
        "ll_shortfall_deg": round(max(mm - 9.0, 0.0), 3),  # LL increase to reach PI−9°
    }


def pi_magnitude_category(pi: float) -> str:
    """Coarse PI band used to set lordosis expectations: low (<45°), average
    (45–60°), high (>60°). A high PI demands a larger lordosis to stay balanced."""
    if pi < 45.0:
        return "low"
    return "average" if pi <= 60.0 else "high"


def roussouly_type_from_ss(ss: float) -> str:
    """Roussouly sagittal morphotype estimated from sacral slope. SS alone cannot
    separate type 1 from 2 (that needs the lordosis apex / segment count), so SS<35
    is reported as "1-2":
        SS < 35° -> "1-2" (low SS: short or flat lordosis)
        35–45°   -> "3"   (harmonious)
        SS > 45° -> "4"   (high SS, long deep lordosis)
    """
    if ss < 35.0:
        return "1-2"
    return "3" if ss <= 45.0 else "4"


def ll_increase_needed(pi: float, ll: float, pt: float) -> float:
    """Greenberg Eq. 73.1 — recommended increase in lumbar lordosis:
    ΔLL ≈ (PI − LL − 9°) + (PT − 20°). Applies when LL is >9° below PI and PT>20°;
    each term is clamped at 0 so it degrades gracefully outside that regime."""
    return round(max(pi - ll - 9.0, 0.0) + max(pt - 20.0, 0.0), 3)


def schwab_sagittal_modifiers(pi: float, ll: float, pt: float,
                              sva_cm: Optional[float] = None) -> Dict:
    """Full SRS-Schwab sagittal grading + Greenberg alignment objectives for one
    case. PT modifier: 0 (<20°), + (20–30°), ++ (>30°). SVA modifier: 0 (<4cm),
    + (4–9.5cm), ++ (>9.5cm) — only if `sva_cm` is supplied (out of scope on v3)."""
    mm = pi - ll
    return {
        "PI-LL": _schwab_grade(abs(mm), 10.0, 20.0),
        "PT": _schwab_grade(pt, 20.0, 30.0),
        "SVA": _schwab_grade(sva_cm, 4.0, 9.5) if sva_cm is not None else "out_of_scope",
        "objectives": {
            "LL=PI±9°": abs(mm) <= 9.0,
            "PT<20°": pt < 20.0,
            "SVA<5cm": (sva_cm < 5.0) if sva_cm is not None else None,
        },
        "ll_increase_needed_deg": ll_increase_needed(pi, ll, pt),
    }


# Lordosis obtainable per technique (Greenberg Table 73.2 + §73.7.3), degrees.
# These are the published ceilings the recommendation is reasoned against.
LORDOSIS_BY_TECHNIQUE = {
    "TLIF/PLIF": 2,        # <0 (kyphosis) up to 2°
    "LLIF": 1,             # XLIF/DLIF/OLIF — indirect, modest
    "ALIF": 6,             # best at L5–S1
    "SPO": 10,             # Smith-Petersen, ~1°/mm bone resected, per level
    "ACR": 12,             # anterior column release, per level (+SVA up to 3 cm)
    "SPO+ACR": 16,
    "PSO": 35,             # pedicle subtraction osteotomy, 30–40°/level
}


def surgical_recommendation(pi: float, ll: float, pt: float) -> Dict:
    """Recommend a lordosis-restoring strategy from the amount of correction needed,
    reasoned ONLY from Greenberg Ch.73 (Eq. 73.1, Table 73.2, Table 73.3, §73.7.3).

    Step 1 — how much lordosis to restore: ΔLL = (PI−LL−9°)+(PT−20°) (Eq. 73.1).
    Step 2 — severity (SRS-Schwab, Table 73.3) from |PI−LL| and PT.
    Step 3 — pick the least-invasive technique whose published lordosis ceiling
             (Table 73.2) covers ΔLL; osteotomy is reserved for large corrections.
    Returns a structured plan (degrees, severity, primary procedure, fixation,
    osteotomy, and a chapter-grounded rationale)."""
    dLL = ll_increase_needed(pi, ll, pt)               # Eq. 73.1
    pill = abs(pi - ll)

    # severity — Table 73.3 (mild / moderate / severe), SRS-Schwab PI–LL & PT
    if pill > 30.0 or pt > 30.0:
        severity = "severe"
    elif pill > 20.0 or pt > 25.0:
        severity = "moderate"
    else:
        severity = "mild"

    # primary lordosis technique by the amount to restore (Table 73.2 ceilings)
    if dLL < 2.0:
        primary = "no major realignment — treat the symptomatic pathology " \
                  "(decompression ± single-level interbody for stability)"
        osteotomy = None
    elif dLL <= LORDOSIS_BY_TECHNIQUE["ALIF"]:
        primary = "anterior/interbody fusion — single- or two-level ALIF " \
                  "(≈6°, best at L5–S1) ± LLIF"
        osteotomy = None
    elif dLL <= LORDOSIS_BY_TECHNIQUE["ACR"]:
        primary = "anterior column release (ACR ≈12°/level) with interbody cage"
        osteotomy = "ACR (anterior, ALL release)"
    elif dLL <= LORDOSIS_BY_TECHNIQUE["SPO+ACR"]:
        primary = "Smith-Petersen osteotomy + ACR (≈16°)"
        osteotomy = "SPO + ACR"
    else:
        n = max(1, int(round(dLL / LORDOSIS_BY_TECHNIQUE["PSO"])))
        primary = f"pedicle subtraction osteotomy (PSO ≈30–40°/level" \
                  + (f", ×{n} levels" if n > 1 else "") + ")"
        osteotomy = "PSO"

    # posterior fixation / standalone (Table 73.3 + §73.7.3): standalone interbody
    # is only an option when PT<20° (well-compensated) with good bone and a ≥22 mm cage
    if pt < 20.0 and dLL <= LORDOSIS_BY_TECHNIQUE["ALIF"]:
        fixation = "standalone interbody feasible (PT<20°, good bone, cage ≥22 mm)"
    elif severity == "severe":
        fixation = "open posterior fixation to S2/ilium ± osteotomy"
    else:
        fixation = "percutaneous posterior fixation (PT≥20°)"

    return {
        "ll_to_restore_deg": dLL,
        "severity": severity,
        "primary": primary,
        "osteotomy": osteotomy,
        "fixation": fixation,
        "objectives": {
            "LL=PI±9°": pill <= 9.0,
            "PT<20°": pt < 20.0,
        },
        "rationale": (
            f"Correction need ΔLL = (PI−LL−9) + (PT−20) = {dLL:.1f}° (Eq. 73.1); "
            f"|PI−LL| {pill:.1f}°, PT {pt:.1f}° → {severity} deformity (Table 73.3). "
            f"Matched to the least-invasive technique whose Table 73.2 lordosis "
            f"ceiling covers {dLL:.1f}°."
        ),
    }


def _infer_mode(data) -> str:
    """Auto-detect '2d' vs '3d' from the input: a landmark dict or a 2-D array → 2d, a
    3-D label volume → 3d."""
    if isinstance(data, dict):
        return "2d"
    arr = np.asarray(data)
    if arr.ndim == 2:
        return "2d"
    if arr.ndim == 3:
        return "3d"
    raise ValueError(f"cannot infer 2d/3d from data with ndim={arr.ndim}")


def spinopelvic_summary(data, affine=None, *, mode: str = None, femoral=None,
                        sup=(0.0, 1.0), case_id: str = "", sup_axis=WORLD_SUPERIOR) -> Dict:
    """Dimension-aware entry point. Routes a 3-D label volume (+ affine) to the 3-D pipeline,
    or a 2-D radiograph — a label mask OR a per-level endplate-line dict (+ optional femoral
    point) — to the 2-D pipeline (ostk.metrics2d). `mode` ('2d'/'3d') forces a backend for
    testing/override; None auto-detects. Both backends return the same summary schema.

    NOTE: the PACS demo calls `spinopelvic_summary_from_label` directly — this router is for
    the CT/radiograph dual-modality path, not the demo."""
    from . import metrics2d
    m = (mode or _infer_mode(data)).lower()
    if m == "2d":
        if isinstance(data, dict):
            return metrics2d.spinopelvic_summary_2d(data, femoral, sup=sup, case_id=case_id)
        return metrics2d.spinopelvic_summary_from_mask_2d(data, sup=sup, case_id=case_id)
    if m == "3d":
        if affine is None:
            raise ValueError("3-D mode requires an affine")
        return spinopelvic_summary_from_label(data, affine, case_id=case_id, sup_axis=sup_axis)
    raise ValueError(f"unknown mode {mode!r} (use '2d', '3d', or None)")


def spinopelvic_summary_from_label(label, affine, *, case_id: str = "",
                                   sup_axis=WORLD_SUPERIOR,
                                   endplate_frac: float = 0.15,
                                   head_frac: float = 0.35,
                                   min_voxels: int = 30) -> Dict:
    """One-call clinical summary of every Greenberg §73 spinopelvic parameter
    computable from a v3 (Vert + S1 + femur) label: PI / SS / PT (PI valid on
    supine CT; SS/PT supine surrogates), LL, PI−LL mismatch, and the SRS-Schwab
    sagittal modifiers + alignment objectives (Eq. 73.1 LL-increase). SVA/TPA are
    omitted — out of scope without C7/T1 (SPEC §5). Returns a JSON-serialisable
    dict; values are None where their inputs were unavailable (flagged, never
    silently dropped)."""
    pi_r, pi_flags = _pi_from_label_core(label, affine, sup_axis, endplate_frac,
                                         head_frac, min_voxels)
    ll_m = lumbar_lordosis_from_label(
        label, affine, case_id=case_id, sup_axis=sup_axis,
        endplate_frac=endplate_frac, head_frac=head_frac, min_voxels=min_voxels)

    PI = round(pi_r["PI"], 3) if pi_r else None
    SS = round(pi_r["SS"], 3) if pi_r else None
    PT = round(pi_r["PT"], 3) if pi_r else None
    LL = ll_m.value

    flags = []
    if pi_r is None:
        flags += [f"PI:{f}" for f in pi_flags]
    if LL is None:
        flags += [f"LL:{f}" for f in ll_m.qc_flags]

    out: Dict = {
        "case_id": case_id, "supine_ct": True,
        "PI": PI, "SS": SS, "PT": PT, "LL": LL,
        "PI-LL": None, "schwab": None,
        "qc_flags": flags or ["ok"],
        "method_version": f"{PI_METHOD_VERSION}+{LL_METHOD_VERSION}",
    }
    if PI is not None:
        out["pi_category"] = pi_magnitude_category(PI)
    if SS is not None:
        out["roussouly"] = roussouly_type_from_ss(SS)
    if PI is not None and LL is not None:
        out["PI-LL"] = pi_ll_mismatch(PI, LL)
        out["schwab"] = schwab_sagittal_modifiers(PI, LL, PT)
        if PT is not None:
            out["surgery"] = surgical_recommendation(PI, LL, PT)
    return out
