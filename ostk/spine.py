"""ostk.spine — vertebral-body endplate fitting (a reusable primitive).

The superior/inferior endplate is the disc-bearing surface of the vertebral
BODY. Two facts make a naive fit wrong:

  * Posterior elements (canal, facets, spinous/transverse processes) are one
    connected component with the body in a 3-D mask, so they can't be split off
    by connectivity — they must be dropped by ANTERIOR position.
  * The endplate is tilted (sacral slope, wedging), so a flat "top-N% by height"
    slab under-reads the tilt. The true face is the extreme voxel per in-plane
    column along the cranio-caudal axis.

`fit_endplate` handles both and returns a plane (centroid, cranial unit normal,
rms). It's used by `ostk.metrics` (lumbar lordosis) and the demo exporter, and is
the place to improve endplate fitting for the whole toolbox.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import WORLD_SUPERIOR, fit_plane_tls, unit


def anterior_axis(normal_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Unit anterior axis: in the sagittal plane (⊥ L–R and ⊥ cranial), oriented
    to world +Y (RAS anterior)."""
    ap = unit(np.cross(np.asarray(lr, float), unit(normal_axis)))
    return ap if ap @ np.array([0.0, 1.0, 0.0]) >= 0 else -ap


def endplate_surface(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                     ap_band=(0.3, 0.9), lat_frac: float = 0.55, nbins: int = 22,
                     lr=(1.0, 0.0, 0.0)) -> np.ndarray:
    """The endplate face of a body point cloud (N,3 world mm): keep the central
    `lat_frac` in L–R (drops the lateral sacral alae / transverse processes) and a
    central ANTERIOR band `ap_band` (quantiles along the anterior axis) — the lower
    bound drops posterior elements, the UPPER bound drops the anterior osteophyte
    lip that otherwise tilts the fit (the L1 failure mode). Then take the extreme
    voxel per in-plane column along `normal_axis` (topmost for 'superior')."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return P
    a = unit(normal_axis)
    lrv = unit(lr)
    if 0.0 < lat_frac < 1.0:                            # central medial band only
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
        if len(P) == 0:
            return P
    ap_lo, ap_hi = ap_band
    if 0.0 <= ap_lo < ap_hi <= 1.0 and (ap_lo > 0.0 or ap_hi < 1.0):
        proj = (P - P.mean(0)) @ anterior_axis(a, lr)
        lo, hi = np.quantile(proj, [ap_lo, ap_hi])
        P = P[(proj >= lo) & (proj <= hi)]
        if len(P) == 0:
            return P
    ref = np.array([1.0, 0, 0]) if abs(a @ np.array([1.0, 0, 0])) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(ref - (ref @ a) * a)
    e2 = np.cross(a, e1)
    u, v, w = P @ e1, P @ e2, P @ a
    ui = np.floor((u - u.min()) / (np.ptp(u) + 1e-9) * nbins).astype(int)
    vi = np.floor((v - v.min()) / (np.ptp(v) + 1e-9) * nbins).astype(int)
    key = ui * (nbins + 1) + vi
    sgn = -1.0 if which == "superior" else 1.0          # superior -> max w first
    order = np.lexsort((sgn * w, key))
    sk = key[order]
    first = np.ones(len(order), bool)
    first[1:] = sk[1:] != sk[:-1]
    return P[order[first]]


def endplate_corners(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                     lat_frac: float = 0.55, drop_post: float = 0.30,
                     ant_skip: float = 0.08, corner_win: float = 0.15,
                     nbins: int = 26, lr=(1.0, 0.0, 0.0)):
    """The two cortical CORNERS that define the clinical AP-corner + tangent endplate
    line (methods 1+3), found body-first and osteophyte-robustly:

      1. medial band (drops lateral processes / sacral alae);
      2. cortical top-profile (highest disc-facing point per A-P column);
      3. BODY: drop the posterior `drop_post` of the A-P extent (pedicle / canal /
         spinous process / dorsal sacrum) BY POSITION — not by height, which would
         chop the low anterior corner of a tilted sacral endplate;
      4. corners: posterior corner at the back of the body; anterior corner a little
         INSIDE the margin (`ant_skip` past the tip — the tangent variant) so an
         anterior osteophyte lip or a margin edge artifact is bridged, not chased.

    Returns (anterior_corner, posterior_corner, body) or None. The chord through the
    corners bridges endplate concavity (standard anatomy) and works on both concave
    (lumbar) and convex (sacral promontory) endplates."""
    P = np.asarray(points, dtype=np.float64)
    a = unit(normal_axis)
    lrv = unit(lr)
    ap = anterior_axis(a, lr)
    if 0.0 < lat_frac < 1.0:
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
    if len(P) < 6:
        return None
    sgn = 1.0 if which == "superior" else -1.0          # which cortical face
    # cortical top-profile: the highest (disc-facing) point per A-P column
    pr = (P - P.mean(0)) @ ap
    edges = np.linspace(pr.min(), pr.max(), nbins + 1)
    prof = []
    for i in range(nbins):
        m = (pr >= edges[i]) & (pr <= edges[i + 1])
        if m.sum() < 4:
            continue
        seg = P[m]
        prof.append(seg[np.argmax(sgn * (seg @ a))])
    prof = np.asarray(prof)
    if len(prof) < 4:
        return None
    # BODY: drop the posterior `drop_post` BY A-P POSITION (pedicle / canal / spinous
    # process / dorsal sacrum) — not by height, which would chop the low anterior
    # corner of a tilted sacral endplate and read the sacral slope as ~flat.
    ppr = (prof - prof.mean(0)) @ ap
    ppr = (ppr - ppr.min()) / (np.ptp(ppr) + 1e-9)      # 0 = posterior, 1 = anterior
    body = prof[ppr >= drop_post]
    bpr = ppr[ppr >= drop_post]
    if len(body) < 3:
        body, bpr = prof, ppr
    bpr = (bpr - bpr.min()) / (np.ptp(bpr) + 1e-9)
    # Tangent corners (clinical 1+3): posterior corner at the back of the body, and
    # the anterior corner a little INSIDE the margin (`ant_skip` past the tip) so an
    # anterior osteophyte lip / margin edge artifact is bridged, not chased.
    post = body[bpr <= corner_win]
    ant = body[(bpr >= 1 - ant_skip - corner_win) & (bpr <= 1 - ant_skip)]
    if len(post) == 0:
        post = body[[int(np.argmin(bpr))]]
    if len(ant) == 0:
        ant = body[[int(np.argmax(bpr))]]
    Pc = post[np.argmax(sgn * (post @ a))]
    A = ant[np.argmax(sgn * (ant @ a))]
    return A, Pc, body


def fit_endplate(points, normal_axis=WORLD_SUPERIOR, which: str = "superior",
                 method: str = "corner", ap_band=(0.3, 0.9), lat_frac: float = 0.55,
                 lr=(1.0, 0.0, 0.0), min_points: int = 30
                 ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Fit the superior/inferior endplate plane of a vertebral-body point cloud.
    Returns (centroid, unit normal oriented cranially for 'superior', rms) or None.

    `method='corner'` (default) is the clinical AP-corner + tangent method: the
    endplate line runs through the anterior- and posterior-superior cortical
    corners, BRIDGING endplate concavity (standard anatomy) the way a radiologist
    draws a Cobb line. `method='surface'` is the biomechanical best-fit to the
    cortical top-surface (least-squares) — truer to the whole surface area but
    pulled into the concavity, so it is not used for sagittal-alignment angles."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < min_points:
        return None
    if method == "corner":
        res = endplate_corners(P, normal_axis, which, lat_frac=lat_frac, lr=lr)
        if res is None:
            return None
        A, Pc, body = res
        mid = 0.5 * (A + Pc)
        n = unit(np.cross(unit(lr), unit(Pc - A)))
        rms = float(np.sqrt(np.mean(((body - mid) @ n) ** 2)))
        a = unit(normal_axis)
        if (which == "superior") != (n @ a >= 0):
            n = -n
        return mid, n, rms
    surf = endplate_surface(P, normal_axis, which, ap_band, lat_frac, lr=lr)
    if len(surf) < min_points:
        surf = P
    c, n, rms = fit_plane_tls(surf)
    # Iteratively reject outliers (MAD-based) so the plane converges to the
    # dominant FLAT endplate: discards anterior osteophyte lips (high outliers)
    # and the posterior down-slope toward the canal/ala (low outliers).
    for _ in range(6):
        d = np.abs((surf - c) @ n)
        thr = 2.0 * np.median(d) + 1e-6
        keep = d <= thr
        if keep.all() or keep.sum() < min_points:
            break
        surf = surf[keep]
        c, n, rms = fit_plane_tls(surf)
    a = unit(normal_axis)
    cranial = n @ a >= 0
    if (which == "superior") != cranial:
        n = -n
    return c, n, rms


def endplate_from_label(label, affine, level: str, which: str = "superior",
                        normal_axis=WORLD_SUPERIOR, method: str = "corner",
                        ap_band=(0.3, 0.9), lat_frac: float = 0.55,
                        lr=(1.0, 0.0, 0.0), min_points: int = 30):
    """Convenience: fit an endplate straight from a label volume + structure name.
    For S1 falls back to the sacrum label if the carved S1 is absent."""
    from .labels import lid
    from .masks import binary_mask, largest_component, mask_world
    m = binary_mask(label, lid(level))
    if level == "S1" and not m.any():
        m = binary_mask(label, lid("sacrum"))
    pts = mask_world(largest_component(m), affine)
    return fit_endplate(pts, normal_axis, which, method=method, ap_band=ap_band,
                        lat_frac=lat_frac, lr=lr, min_points=min_points)
