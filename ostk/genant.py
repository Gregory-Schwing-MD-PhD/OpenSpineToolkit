"""ostk.genant -- six-point vertebral morphometry, to the published convention.

WHAT THIS IMPLEMENTS. Genant's semiquantitative scheme (1993) and the quantitative
morphometry built on it are the reference method for vertebral deformity, and both rest on
SIX points per vertebra on a lateral view:

        superior:   anterior (sa)    middle (sm)    posterior (sp)
        inferior:   anterior (ia)    middle (im)    posterior (ip)

from which three heights follow -- anterior Ha = |sa-ia|, middle Hm = |sm-im|, posterior
Hp = |sp-ip| -- and the grade from their ratios.

THE CONVENTION IS THE HARD PART, NOT THE ARITHMETIC. Three rules carry almost all of the
difficulty, and all three are decisions a reader makes rather than geometry a program can
discover:

  PRE-OSTEOPHYTE. Corners mark "the corners of the vertebral bodies PRIOR TO ANY OSTEOPHYTE
  FORMATION" (Hipp et al., JBMR Plus 2022;6:e10677, applying Genant). An anterior spur can
  extend the silhouette several millimetres, and a corner taken as the extreme point of a
  mask or a contour follows it. That is the single most common way an automated corner
  disagrees with a human one, and it disagrees WORST on the degenerative spines where the
  measurement matters most.

  MIDSAGITTAL, NOT SILHOUETTE. On a lateral radiograph the left and right endplate rims
  project as a double contour. The landmark BISECTS that pair; it is not the outer edge.
  Hipp states the intent plainly: the posterior superior landmark is placed "to represent
  the endplate as it would appear on a midsagittal slice of a CT exam".

  THE MIDDLE POINT IS A CONSTRUCTION. sm and im are not anatomical features; they are the
  midpoint of the endplate between the two corners, and they are where automated methods are
  least accurate. The one published two-stage system reports landmark error rising "in more
  severely collapsed vertebrae, particularly at central landmarks" (JBMR Plus 2025;9:ziaf017).

SO THIS MODULE DOES NOT TRY TO DISCOVER CORNERS FROM A MASK. Where corners are supplied --
by a reader, or by a keypoint model trained on readers -- it derives the six points, the
three heights, the ratios and the grade, to the published definitions. Where only a
projected body outline is available it offers a plane-intersection estimate, and says in its
own return value that the estimate is post-osteophyte and therefore not the convention.

    from ostk.genant import six_points, heights, ratios, genant_grade
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SixPoints",
    "six_points",
    "heights",
    "ratios",
    "genant_grade",
    "GRADE_BOUNDS",
    "wedge_angle",
]

# Genant's bands, on the largest relative height reduction against the vertebra's own
# reference height. The 0.5 "borderline" band is from the later quantitative literature
# rather than the 1993 paper, and is reported separately so a caller can ignore it.
GRADE_BOUNDS = (
    (0.15, 0.5),    # 15-20%   borderline
    (0.20, 1.0),    # 20-25%   mild
    (0.25, 2.0),    # 25-40%   moderate
    (0.40, 3.0),    # >40%     severe
)


@dataclass
class SixPoints:
    """The six landmarks, in image or world units, plus what they are trusted for."""

    sa: np.ndarray
    sm: np.ndarray
    sp: np.ndarray
    ia: np.ndarray
    im: np.ndarray
    ip: np.ndarray
    #: False when the middle points were constructed rather than placed by a reader.
    middles_observed: bool = False
    #: True only when the corners follow the pre-osteophyte convention -- which a program
    #: cannot establish for itself. Callers that derive corners from a mask must leave this
    #: False, and anything comparing against published norms should check it.
    pre_osteophyte: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def as_array(self) -> np.ndarray:
        """(6, d) in the canonical order sa, sm, sp, ia, im, ip."""
        return np.vstack([self.sa, self.sm, self.sp, self.ia, self.im, self.ip])


def _v(p) -> np.ndarray:
    a = np.asarray(p, dtype=float).ravel()
    if a.size < 2:
        raise ValueError(f"landmark needs at least two components, got {a.size}")
    return a


def six_points(sup_ant, sup_post, inf_ant, inf_post, *,
               sup_mid=None, inf_mid=None,
               pre_osteophyte: bool = False) -> SixPoints:
    """Assemble the six points from four corners, constructing the middles if absent.

    The four corners are the usual output of a keypoint model. Genant needs six, and the
    two extra are the midpoints of each endplate -- a construction, not a feature, which is
    why they are marked `middles_observed=False` when built here. A model trained on
    six-point annotations should pass its own sm and im instead: on a collapsed body the
    endplate is concave and its true midpoint sits BELOW the chord between the corners, so
    the constructed point overestimates middle height exactly where middle height is the
    measurement of interest.

    `pre_osteophyte` is the caller's assertion about the corners it is handing in. It is
    never inferred, because it cannot be.
    """
    sa, sp = _v(sup_ant), _v(sup_post)
    ia, ip = _v(inf_ant), _v(inf_post)
    notes = []

    if sup_mid is None:
        sm = 0.5 * (sa + sp)
        notes.append("superior middle constructed as the corner midpoint; on a concave "
                     "endplate this sits above the true midpoint and overestimates Hm")
        observed = False
    else:
        sm = _v(sup_mid)
        observed = True

    if inf_mid is None:
        im = 0.5 * (ia + ip)
        if observed:
            notes.append("inferior middle constructed while the superior one was observed")
        observed = False
    else:
        im = _v(inf_mid)

    if not pre_osteophyte:
        notes.append("corners not asserted pre-osteophyte: heights may follow a spur "
                     "rather than the body, and should not be compared to Genant norms")

    return SixPoints(sa=sa, sm=sm, sp=sp, ia=ia, im=im, ip=ip,
                     middles_observed=observed, pre_osteophyte=pre_osteophyte,
                     notes=tuple(notes))


def heights(pts: SixPoints) -> Dict[str, float]:
    """Anterior, middle and posterior heights, as plain Euclidean distances.

    NOT vertical extents. A wedged or rotated vertebra has endplates that are not
    horizontal, and taking a difference in y would shorten every height by the cosine of
    that tilt -- which is largest on the deformed vertebrae the measure exists to find.
    """
    return {
        "Ha": float(np.linalg.norm(pts.sa - pts.ia)),
        "Hm": float(np.linalg.norm(pts.sm - pts.im)),
        "Hp": float(np.linalg.norm(pts.sp - pts.ip)),
    }


def ratios(pts: SixPoints) -> Dict[str, float]:
    """The three published ratios. Hp is the usual reference height.

    Returned as measured; a caller comparing against norms should also check
    `pts.pre_osteophyte` and `pts.middles_observed`.
    """
    h = heights(pts)
    out: Dict[str, float] = {}
    if h["Hp"] > 0:
        out["Ha_Hp"] = h["Ha"] / h["Hp"]       # wedge
        out["Hm_Hp"] = h["Hm"] / h["Hp"]       # biconcavity
    if h["Ha"] > 0:
        out["Hm_Ha"] = h["Hm"] / h["Ha"]
    return out


def wedge_angle(pts: SixPoints) -> Optional[float]:
    """Angle between the superior and inferior end-plate chords, in degrees.

    Reported alongside the ratios because they disagree in an informative way: a uniformly
    compressed body has ratios near one and a wedged one does not, while a body that is
    wedged without losing height shows in the angle and not in Ha/Hp.
    """
    u = pts.sp - pts.sa
    v = pts.ip - pts.ia
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return None
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    a = float(np.degrees(np.arccos(c)))
    return a if a <= 90.0 else 180.0 - a


def genant_grade(pts: SixPoints, *, reference: Optional[float] = None,
                 include_borderline: bool = False) -> Dict[str, object]:
    """Grade from the largest relative height reduction.

    `reference` is the height a normal vertebra at this level would have. Genant's scheme
    compares a vertebra to its NEIGHBOURS or to an expected height, not to itself; without
    a reference the only available comparison is the vertebra's own posterior height, which
    is blind to a vertebra compressed uniformly at all three points. That limitation is
    returned rather than hidden.

    The grade is advisory. Genant's method is semiQUANTITATIVE by design: the reader's
    judgement of whether a short vertebra is fractured or merely small is part of it, and
    no ratio reproduces that.
    """
    h = heights(pts)
    ref = float(reference) if reference else h["Hp"]
    result: Dict[str, object] = {"heights": h, "ratios": ratios(pts),
                                 "wedge_angle_deg": wedge_angle(pts),
                                 "reference_height": ref,
                                 "reference_is_own_Hp": reference is None}
    if ref <= 0:
        result["grade"] = None
        result["why"] = "no usable reference height"
        return result

    losses = {k: max(0.0, 1.0 - v / ref) for k, v in h.items()}
    worst_key = max(losses, key=losses.get)
    worst = losses[worst_key]

    grade = 0.0
    for thresh, g in GRADE_BOUNDS:
        if worst >= thresh:
            grade = g
    if grade == 0.5 and not include_borderline:
        grade = 0.0

    result.update({
        "grade": grade,
        "worst_loss": worst,
        "worst_height": worst_key,
        "caveats": tuple(pts.notes) + (
            () if pts.pre_osteophyte else
            ("corners are not asserted pre-osteophyte",)) + (
            () if pts.middles_observed else
            ("middle points were constructed, so Hm is the least reliable height",)) + (
            ("reference is the vertebra's own posterior height, so a uniformly compressed "
             "body grades 0",) if reference is None else ()),
    })
    return result
