"""ostk.metrics2d — spinopelvic parameters in a 2-D sagittal plane (e.g. a lateral
radiograph), mirroring ostk.metrics one-for-one.

The 3-D pipeline measures every angle *in the sagittal plane* (it projects out the
L–R axis first). A lateral radiograph **is** that plane already, so the 2-D port uses
the IDENTICAL definitions on 2-D vectors — the angle a parameter returns is therefore
the same number the 3-D code returns for the same geometry (see the cross-checks in
tests/test_metrics2d.py). PI needs the femoral-head point; without it (e.g. a lumbar
film that crops the hips) PI/PT are returned as None and LL/SS still compute.

Convention: 2-D points are (x, y); `sup` is the superior direction in that frame
(default (0, 1) = +y up). Endplates are given as a LINE = (corner_a, corner_b)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .geometry import unit, angle_between
from .metrics import (pi_ll_mismatch, schwab_sagittal_modifiers, roussouly_type_from_ss,
                      pi_magnitude_category, surgical_recommendation, LL_ENDPLATE_CHAIN)

Pt = Sequence[float]
Line = Sequence[Pt]


def _perp(v) -> np.ndarray:
    """In-plane perpendicular (rotate +90°)."""
    v = unit(v)
    return np.array([-v[1], v[0]])


def _long_axis_2d(points) -> np.ndarray:
    P = np.asarray(points, float)
    C = np.cov((P - P.mean(0)).T)
    w, V = np.linalg.eigh(C)
    return unit(V[:, int(np.argmax(w))])


def fit_circle_2d(points):
    """Algebraic least-squares circle (Kåsa), the 2-D analogue of geometry.fit_sphere —
    robust to a partial arc (FOV-clipped femoral head). Returns (center, radius, rms)."""
    P = np.asarray(points, float)
    A = np.c_[2.0 * P, np.ones(len(P))]
    b = np.sum(P ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    r = float(np.sqrt(max(sol[2] + c @ c, 0.0)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return c, r, rms


def _endplate(line: Line, sup) -> Dict:
    """Return {dir, normal(superior-oriented), mid} for an endplate line."""
    a, b = np.asarray(line[0], float), np.asarray(line[1], float)
    e = unit(b - a)
    n = _perp(e)
    if n @ unit(sup) < 0:
        n = -n
    return {"dir": e, "normal": n, "mid": 0.5 * (a + b)}


# ---- the four parameters (identical definitions to ostk.metrics) ------------
def cobb_2d(line_a: Line, line_b: Line) -> float:
    """Acute angle (deg) between two endplate lines — the 2-D Cobb (cf. geometry.cobb_angle,
    which takes the acute dihedral of the two endplate normals in the viewing plane)."""
    a, b = _endplate(line_a, (0, 1))["dir"], _endplate(line_b, (0, 1))["dir"]
    return float(np.degrees(np.arccos(abs(float(np.clip(a @ b, -1.0, 1.0))))))


def sacral_slope_2d(s1_line: Line, sup=(0, 1)) -> float:
    """SS — angle between the S1 endplate and the horizontal, computed (like the 3-D code)
    as the angle between the cranially-oriented endplate NORMAL and vertical, so it is
    always the 0–90° acute value regardless of corner order."""
    return angle_between(_endplate(s1_line, sup)["normal"], unit(sup))


def pelvic_tilt_2d(s1_line: Line, femoral: Pt, sup=(0, 1)) -> float:
    """PT — angle between vertical and the femoral-head→S1-midpoint radius."""
    P = _endplate(s1_line, sup)["mid"]
    return angle_between(P - np.asarray(femoral, float), unit(sup))


def pelvic_incidence_2d(s1_line: Line, femoral: Pt, sup=(0, 1)) -> float:
    """PI — angle between the S1-endplate perpendicular and the radius (= SS + PT)."""
    ep = _endplate(s1_line, sup)
    return angle_between(ep["normal"], ep["mid"] - np.asarray(femoral, float))


def lumbar_lordosis_2d(endplates: Dict[str, Line]) -> Dict:
    """LL — Cobb between the most-cranial and most-caudal present lumbar endplates
    (matches ostk.metrics.lumbar_lordosis: top↔bot Cobb, not a per-segment sum)."""
    present = [lv for lv in LL_ENDPLATE_CHAIN if lv in endplates]
    if len(present) < 2:
        return {"LL": None, "levels": present}
    top, bot = present[0], present[-1]
    return {"LL": round(cobb_2d(endplates[top], endplates[bot]), 3),
            "levels": present, "span": f"{top}-{bot}"}


# ---- summary (same schema as metrics.spinopelvic_summary_from_label) --------
def spinopelvic_summary_2d(endplates: Dict[str, Line], femoral: Optional[Pt] = None,
                           *, sup=(0, 1), case_id: str = "") -> Dict:
    """One-call 2-D spinopelvic summary from per-level SUPERIOR endplate lines (+ the
    femoral-head point for PI/PT). Returns the SAME dict schema as the 3-D summary so the
    two are drop-in comparable; values are None where inputs were unavailable."""
    sup = unit(sup)
    ll = lumbar_lordosis_2d(endplates)
    LL = ll["LL"]
    s1 = endplates.get("S1")
    PI = SS = PT = None
    flags = []
    if s1 is not None:
        SS = round(sacral_slope_2d(s1, sup), 3)
        if femoral is not None:
            PI = round(pelvic_incidence_2d(s1, femoral, sup), 3)
            PT = round(pelvic_tilt_2d(s1, femoral, sup), 3)
        else:
            flags.append("no_femoral_head:PI/PT_skipped")
    else:
        flags.append("no_S1_endplate")
    if LL is None:
        flags.append("LL:<2_endplates")

    out: Dict = {
        "case_id": case_id, "supine_ct": False, "modality": "radiograph_2d",
        "PI": PI, "SS": SS, "PT": PT, "LL": LL,
        "PI-LL": None, "schwab": None,
        "qc_flags": flags or ["ok"],
        "method_version": "2d-1",
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


# ---- 2-D LABEL MASK -> endplate lines (so a 2-D seg routes like the 3-D one) -
def endplates_from_mask_2d(mask, *, sup=(0, 1), endplate_frac: float = 0.30,
                           min_pixels: int = 20):
    """Extract per-vertebra SUPERIOR endplate lines + the femoral-head point from a 2-D
    label mask (same ostk label ids as 3-D). The superior endplate = a line fit through
    the top `endplate_frac` slab of each body (the 2-D analogue of the 3-D slab + plane
    fit). Returns (endplates_dict, femoral_point_or_None)."""
    from .labels import LABELS
    sup = unit(sup)
    m = np.asarray(mask)
    endplates: Dict[str, Line] = {}
    for name in LL_ENDPLATE_CHAIN:
        lid = LABELS.get(name)
        if lid is None:
            continue
        pts = np.argwhere(m == lid)[:, ::-1].astype(float)   # (col,row)=(x,y)
        if len(pts) < min_pixels:
            continue
        s = pts @ sup
        slab = pts[s >= np.quantile(s, 1.0 - endplate_frac)]
        c = slab.mean(0)
        e = _long_axis_2d(slab) if len(slab) >= 3 else _perp(sup)
        half = 0.5 * float((slab @ e).max() - (slab @ e).min()) or 10.0
        endplates[name] = (c - half * e, c + half * e)
    # femoral-head point = midpoint of the two femoral-head circle centers
    fem = []
    for name in ("femur_left", "femur_right", "left_hip", "right_hip"):
        lid = LABELS.get(name)
        if lid is None:
            continue
        pts = np.argwhere(m == lid)[:, ::-1].astype(float)
        if len(pts) >= min_pixels:
            fem.append(fit_circle_2d(pts)[0])
    femoral = (np.mean(fem, axis=0) if fem else None)
    return endplates, femoral


def spinopelvic_summary_from_mask_2d(mask, *, sup=(0, 1), case_id: str = "", **kw) -> Dict:
    """2-D summary straight from a 2-D label mask (extract endplates + femoral, then measure)."""
    endplates, femoral = endplates_from_mask_2d(mask, sup=sup, **kw)
    return spinopelvic_summary_2d(endplates, femoral, sup=sup, case_id=case_id)
