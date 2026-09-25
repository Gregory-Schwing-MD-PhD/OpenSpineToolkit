"""Vertebral BODY isolation and endplate geometry.

Why this module exists
----------------------
Everything that goes wrong when you fit an endplate from a whole-vertebra mask goes
wrong for one reason: the posterior elements are still in the data. The pedicles,
lamina and spinous process sit behind the body, at endplate height, and any fit that
sees them is pulled backwards. Working around that downstream produces exactly the
symptoms we measured -- a posterior corner landing inside the spinal canal, an endplate
line crossing into the next vertebra's arch, and a `drop_post` fraction that cuts at a
different anatomic place on every level because it is a fraction of an extent that
includes the very structures it is meant to exclude.

So isolate the body FIRST. Then the endplate is unambiguous and the downstream
parameters (`drop_post`, rim bands, canal-gap searches, snap limits) are not needed at
all -- there is nothing left to exclude.

Method and prior art
--------------------
1. BODY / POSTERIOR-ELEMENT PARTITION -- the SPINAL CANAL, with watershed as a fallback.

   The canal is a topological HOLE: in an axial section the body, pedicles and lamina
   enclose it. Fill the holes slice by slice and the filled-minus-original difference IS
   the canal, with no threshold to choose. Its anterior boundary is the body's posterior
   wall -- the "posterior vertebral body line" that is read directly off a lateral film.

   A distance-transform watershed was tried first and is kept only as a fallback, because
   it assumes the body/arch junction is a THIN neck. That holds in the thoracic spine and
   fails in the lumbar, where pedicles are 8-18 mm across: measured on case 0004 it left
   L2, L3 and L4 completely unpartitioned (100% of the mask returned as "body"), and the
   endplate spans blew out to 70-92 mm as the posterior corner ran to the spinous process.
   The canal criterion has no such level dependence -- every vertebra encloses a canal.

   Watershed reference, still used when the canal cannot be recovered (an incomplete arch,
   e.g. spondylolysis, leaves no enclosed hole to fill):
   The pedicle isthmus is a thin neck joining two thick regions, which is the canonical
   watershed cut. Markers are the connected components of the mask eroded in distance
   terms (EDT >= `core_mm`), so the body core and the posterior-element core each seed
   one basin, and the flood meets at the isthmus.
     Yao J, O'Connor SD, Summers RM. "Automated spinal column extraction and
       partitioning." IEEE ISBI 2006, 390-393.
     Naegel B. "Using mathematical morphology for the anatomical labeling of vertebrae
       from 3D CT-scan images." Comput Med Imaging Graph 2007;31(3):141-156.
   The body is identified as the basin containing the global distance-transform maximum
   -- the vertebral body is the thickest part of a vertebra, which holds whether or not
   the posterior elements are complete, and needs no anterior axis to be known first.

2. BODY COORDINATE SYSTEM.
     Mastmeyer A, Engelke K, Fuchs C, Kalender WA. "A hierarchical 3D segmentation
       method and the definition of vertebral body coordinate systems for QCT of the
       lumbar spine." Med Image Anal 2006;10(4):560-577.
   Mastmeyer isolate the body for the same reason we do: in QCT the posterior arch
   corrupts the measurement. We keep the patient axes rather than the body's own PCA
   axes, because every angle in ostk is defined against the patient frame and switching
   would silently redefine SS/LL.

3. ENDPLATE PLANE -- RANSAC.
     Fischler MA, Bolles RC. "Random sample consensus: a paradigm for model fitting
       with applications to image analysis and automated cartography."
       Commun ACM 1981;24(6):381-395.
   RANSAC is the standard robust estimator for exactly this failure mode: a majority of
   inliers (the endplate) plus a minority of gross outliers (an anterior osteophyte lip,
   a Schmorl's node, a segmentation spur). It replaces iterative trimming, which has no
   breakdown guarantee and which we measured failing -- residuals of 7-12 mm against
   0.5-2.5 mm on sound levels.

4. CORNERS -- the A-P extremes of the endplate surface.
   Reported as the chord through the two corners, not the RANSAC plane itself, because
   the endplate is CONCAVE and the clinical line bridges that concavity: it is drawn
   corner to corner, the same construction a Cobb angle uses.
     Vrtovec T, Pernus F, Likar B. "A review of methods for quantitative evaluation of
       spinal curvature." Eur Spine J 2009;18(5):593-607.
     Stern D, Likar B, Pernus F, Vrtovec T. "Parametric modelling and segmentation of
       vertebral bodies in 3D CT and MR spine images." Phys Med Biol 2011;56(23):7505-22.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

from .geometry import WORLD_SUPERIOR, unit
from .spine import anterior_axis


def voxel_spacing(affine) -> np.ndarray:
    """Physical voxel size (mm) along each array axis."""
    return np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)


def _sup_array_axis(affine, sup_axis) -> int:
    """Array axis most closely aligned with the patient's superior direction."""
    M = np.asarray(affine, float)[:3, :3]
    return int(np.argmax(np.abs(unit(np.asarray(sup_axis, float)) @ M / np.linalg.norm(M, axis=0))))


def canal_mask(mask, affine, *, sup_axis=WORLD_SUPERIOR, min_vox: int = 20
               ) -> Optional[np.ndarray]:
    """The spinal canal enclosed by this vertebra, as a topological hole.

    Filled per axial section rather than in 3-D: in 3-D the canal is an open tube, so it
    is not a hole at all and nothing would be filled. Returns None when the arch does not
    close the ring in enough sections to be believable -- an incomplete or lytic arch --
    rather than returning a sliver that would put the wall in the wrong place.
    """
    m = np.asarray(mask, bool)
    ax = _sup_array_axis(affine, sup_axis)
    filled = np.zeros_like(m)
    mv = np.moveaxis(m, ax, 0)
    fv = np.moveaxis(filled, ax, 0)
    # CLOSE THE RING BEFORE LOOKING FOR THE HOLE. A lamina one voxel short of meeting
    # leaves the canal open to the background, binary_fill_holes then finds nothing at
    # all, and the section is lost -- which is why so many levels came back with no
    # canal on thick or noisy scans. Banik, Rangayyan & Boag (J Digit Imaging 23:301)
    # close the canal with a tubular element 2 mm in radius for exactly this; the same
    # radius in-plane bridges the interlaminar gap.
    #
    # The closing is used ONLY TO FIND the hole. The hole is then taken against the
    # ORIGINAL mask, so the canal is not shrunk by the bone the closing invented and
    # every downstream use -- the body cut above all -- sees the same wall it always did.
    sp = np.sqrt((np.asarray(affine, float)[:3, :3] ** 2).sum(axis=0))
    in_plane = float(min(v for i, v in enumerate(sp) if i != ax))
    rr = max(1, int(round(2.0 / in_plane)))
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    se = (xx ** 2 + yy ** 2) <= rr ** 2
    mvc = np.moveaxis(m, ax, 0)
    for k in range(mv.shape[0]):
        if mv[k].any():
            fv[k] = ndimage.binary_fill_holes(ndimage.binary_closing(mvc[k], se))
    canal = np.moveaxis(fv, 0, ax) & ~m
    if canal.sum() < min_vox:
        return None
    lab, n = ndimage.label(canal)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def body_mask_by_canal(mask, affine, *, sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                       margin_mm: float = 0.0, canal_from=None) -> Optional[np.ndarray]:
    """BODY = the part of the mask anterior to the canal, cut SECTION BY SECTION.

    The cut follows the posterior vertebral body line, which is CONCAVE -- it bows
    anteriorly at mid-body and runs further posterior at the endplate rims. A single
    plane at the canal's globally anterior-most point therefore cuts at the mid-body
    concavity and shaves the posterior rim off BOTH endplates, which is visible on an
    overlay as a superior corner that does not reach far enough back. So the wall is
    taken per axial section instead, and sections with no canal (above and below its
    extent) inherit the nearest section that has one.

    `canal_from` supplies a LARGER mask to find the canal in, while the cut is still
    applied to `mask`. Needed for S1, which in the v4 scheme is carved off the top of the
    sacrum: that carve can slice the arch so the ring never closes in any one section,
    and S1 alone then yields no canal at all. Measured on two cases -- 0001 S1: 0 closed
    sections, 0 hole voxels; 0004 S1: 6 and 4693. The sacral canal does not stop at an
    artificial label boundary, so pass S1 | sacrum and the ring closes (0001 sacrum: 47
    closed sections). Without this S1 silently took the watershed fallback and the
    endplate line ran across the whole sacrum into the posterior elements.

    Unlike a thickness or fraction rule this is the same anatomic landmark at every
    level, in every patient.
    """
    canal = canal_mask(mask if canal_from is None else canal_from,
                       affine, sup_axis=sup_axis)
    if canal is None:
        return None
    ap = anterior_axis(unit(sup_axis), lr)
    aff = np.asarray(affine, float)
    ax = _sup_array_axis(affine, sup_axis)
    m = np.asarray(mask, bool)

    ci = np.array(np.nonzero(canal)).T
    cw = (np.c_[ci, np.ones(len(ci))] @ aff.T)[:, :3]
    ap_c, k_c = cw @ ap, ci[:, ax]

    n_sl = m.shape[ax]
    # -inf, NOT nan: np.maximum propagates nan, so a nan-initialised accumulator stays
    # nan for every section, `have` comes back empty, and this function silently returns
    # None for EVERY vertebra -- handing all of them to the watershed fallback, which is
    # exactly the level-dependent method this replaces. It fails loudly nowhere and
    # quietly everywhere; the only visible symptom was endplate lines running through the
    # whole sacrum and L4/L5 spans of 57 mm.
    wall = np.full(n_sl, -np.inf)
    np.maximum.at(wall, k_c, ap_c)                    # anterior canal edge per section
    have = np.nonzero(np.isfinite(wall))[0]
    if not len(have):
        return None
    # sections outside the canal's extent inherit the nearest section that has one
    idx = np.clip(np.searchsorted(have, np.arange(n_sl)), 0, len(have) - 1)
    prev = np.clip(idx - 1, 0, len(have) - 1)
    pick = np.where(np.abs(have[idx] - np.arange(n_sl))
                    <= np.abs(have[prev] - np.arange(n_sl)), have[idx], have[prev])
    wall = wall[pick] + float(margin_mm)

    mi = np.array(np.nonzero(m)).T
    mw = (np.c_[mi, np.ones(len(mi))] @ aff.T)[:, :3]
    sel = (mw @ ap) >= wall[mi[:, ax]]
    if sel.sum() < 30:
        return None
    out = np.zeros_like(m)
    ii = mi[sel]
    out[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return out


def body_mask_watershed(mask, affine, *, core_mm: float = 4.0, min_core_vox: int = 30
                        ) -> Optional[np.ndarray]:
    """Split a whole-vertebra mask into body / posterior elements; return the BODY.

    Watershed on the distance transform, markers from the distance-thresholded cores
    (Yao 2006). `core_mm` is a THICKNESS, in mm, not a fraction of anything -- a pedicle
    isthmus is thinner than this and a vertebral body is thicker, at every level and in
    every patient, which is why this generalises where a fixed fraction does not.

    Returns a boolean array, or None if the partition is not possible (a single core, a
    mask too small to be a vertebra). FALLBACK ONLY -- see body_mask.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return None
    sp = voxel_spacing(affine)
    edt = ndimage.distance_transform_edt(m, sampling=sp)
    cores, n = ndimage.label(edt >= float(core_mm))
    if n == 0:
        return None
    # drop specks so a stray thick voxel cannot seed a basin
    sizes = ndimage.sum(np.ones_like(cores), cores, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_core_vox]
    if not keep:
        return None
    if len(keep) == 1:                       # nothing to separate: body only, or fused
        return m
    markers = np.zeros_like(cores)
    for new, old in enumerate(keep, start=1):
        markers[cores == old] = new
    from skimage.segmentation import watershed
    basins = watershed(-edt, markers=markers, mask=m)
    # the BODY is the basin holding the global thickness maximum -- a vertebral body is
    # the thickest part of a vertebra whether or not the arch is complete
    body_label = int(basins[np.unravel_index(np.argmax(edt * (basins > 0)), edt.shape)])
    if body_label == 0:
        return None
    return basins == body_label


def fit_plane_ransac(points, *, thresh_mm: float = 1.5, iters: int = 200,
                     min_inlier_frac: float = 0.5, seed: int = 0
                     ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Robust plane fit by RANSAC (Fischler & Bolles 1981).

    Returns (centroid, unit normal, inlier boolean mask), or None. Unlike iterative
    trimming this has a breakdown point: gross outliers -- an osteophyte lip, a
    segmentation spur -- cannot move the consensus as long as the endplate itself is the
    largest coherent planar set, which it is.
    """
    P = np.asarray(points, float)
    if len(P) < 3:
        return None
    rng = np.random.default_rng(seed)
    n_it = int(iters)
    # VECTORISED over hypotheses. The textbook form -- a Python loop drawing 3 points and
    # calling np.cross per iteration -- spends nearly all its time in numpy's per-call
    # overhead on 3-vectors, not on arithmetic: it was the single hottest line in a whole
    # case build (12k np.cross calls). Drawing every triple at once and scoring them in
    # one broadcast is the identical estimator, just without the interpreter in the loop.
    idx = rng.integers(0, len(P), size=(n_it, 3))
    A, B, Cc = P[idx[:, 0]], P[idx[:, 1]], P[idx[:, 2]]
    nrm = np.cross(B - A, Cc - A)                      # (n_it, 3)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-9
    if not ok.any():
        return None
    nrm = nrm[ok] / ln[ok, None]
    A = A[ok]
    # |(p - a) . n| for every point x every hypothesis  -> (n_pts, n_hyp)
    d = np.abs(P @ nrm.T - np.einsum("ij,ij->i", A, nrm)[None, :])
    inl = d <= thresh_mm
    best_i = int(np.argmax(inl.sum(axis=0)))
    best_in = inl[:, best_i]
    if best_in.sum() < max(3, min_inlier_frac * len(P) * 0.2):
        return None
    # refit on the consensus set by total least squares
    Q = P[best_in]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    return c, unit(vt[-1]), best_in


def endplate_surface_body(body_pts, which: str, *, sup_axis=WORLD_SUPERIOR,
                          lr=(1.0, 0.0, 0.0), nbins: int = 26) -> np.ndarray:
    """Disc-facing cortical surface of an ISOLATED body: the extreme voxel along the
    superior axis in each (A-P, L-R) cell.

    No posterior-element rejection here -- `body_pts` has none. That is the whole point
    of doing the partition first.
    """
    P = np.asarray(body_pts, float)
    if len(P) < 6:
        return P
    a = unit(sup_axis)
    lrv = unit(lr)
    ap = anterior_axis(a, lr)
    sgn = 1.0 if which == "superior" else -1.0
    sc = (P - P.mean(0)) @ ap
    lc = (P - P.mean(0)) @ lrv
    si = np.floor((sc - sc.min()) / (np.ptp(sc) + 1e-9) * nbins).astype(int)
    li = np.floor((lc - lc.min()) / (np.ptp(lc) + 1e-9) * nbins).astype(int)
    key = si * (nbins + 1) + li
    order = np.lexsort((-sgn * (P @ a), key))
    ks = key[order]
    first = np.ones(len(order), bool)
    first[1:] = ks[1:] != ks[:-1]
    return P[order[first]]


def endplate_corners_body(mask, affine, which: str = "superior", *,
                          sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                          core_mm: float = 4.0, ransac_mm: float = 1.5,
                          lat_frac: float = 0.45, min_points: int = 30,
                          canal_from=None, body=None, edge_trim_frac: float = 0.12,
                          plate_band_mm: float = 8.0, profile_degree: int = 2):
    """Endplate corners of ONE vertebra, from its whole-vertebra mask.

    body_mask -> medial band -> endplate surface -> RANSAC plane -> A-P extremes of the
    inliers. Returns (anterior, posterior, rms, n_inliers) in world mm, or None.

    `rms` is the residual of the endplate surface about the RANSAC plane and is the
    honest confidence on this endplate -- a level whose plate did not converge should be
    dropped, not drawn. The medial band is kept (it drops the transverse processes and,
    on the sacrum, the alae, which are part of the body's own component and so survive
    the watershed) but every other tuning parameter of the old path is gone.
    """
    m = np.asarray(mask, bool)
    # pass the CALLER's axes through -- defaulting silently to the world axes made this
    # correct only when the caller happened to use them, which a test harness does and
    # the DRR path (detector cranial axis, beam direction) does not
    # `body` lets the caller isolate ONCE and reuse it for both endplate faces. The
    # isolation (per-section hole fill + the anterior cut) is the expensive step and it is
    # a property of the vertebra, not of which face is being fitted -- recomputing it per
    # face doubled the cost of every case for no change in result.
    if body is None:
        body = body_mask(m, affine, sup_axis=sup_axis, lr=lr, core_mm=core_mm,
                         canal_from=canal_from)
    if body is None or body.sum() < min_points:
        return None
    idx = np.array(np.nonzero(body)).T
    pts = (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]
    lrv = unit(lr)
    if 0.0 < lat_frac < 1.0:
        lp = pts @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        pts = pts[(lp >= lo) & (lp <= hi)]
    if len(pts) < min_points:
        return None
    surf = endplate_surface_body(pts, which, sup_axis=sup_axis, lr=lr)
    if len(surf) < 6:
        return None
    fit = fit_plane_ransac(surf, thresh_mm=ransac_mm)
    if fit is None:
        return None
    c, n, inl = fit
    keep = surf[inl]
    if len(keep) < 4:
        return None
    ap = anterior_axis(unit(sup_axis), lr)
    a_ax = unit(sup_axis)
    # ---- corners = endplate tangent x cortical WALL (Frobin) ------------------------
    # A corner is CONSTRUCTED, not observed. The last voxel on the endplate stops where
    # the cortex begins curving up into the wall, so the posterior corner quits before
    # the body surface turns -- exactly the "not posterior enough" seen on L1-L3. The
    # reader's construction is to run the endplate tangent back until it meets the
    # posterior wall, which stays well defined where the surface itself is not.
    #   Frobin W, Brinckmann P, Biggemann M, Tillotson M, Burton K. Clin Biomech
    #     1997;12(Suppl 1):S1-S63 -- vertebral body as a quadrilateral, corners as the
    #     endplate x wall intersections; the reference method for lateral-radiograph
    #     vertebral morphometry.
    #   Hurxthal LM. Am J Roentgenol 1968;103(3):635-644.
    #   Black DM, Palermo L, Nevitt MC, et al. J Bone Miner Res 1995;10(6):890-902.
    # Osteophytes fall out for free: each line is fitted by RANSAC over its own cortical
    # margin, and a spondylophyte is a gross outlier to both the plate it grows from and
    # the wall it projects past, so it can move neither. Measure the body, ignore the spur.
    #
    # The plate is refit to the RIM first, because a plane fitted to the whole surface is
    # dragged into the endplate's central concavity; the clinical line is the tangent
    # bridging that concavity, the same construction a Cobb line uses.
    # Drop the outermost A-P columns FIRST. Past the edge of the plate the topmost
    # voxel in a column is no longer endplate at all -- it is the cortical WALL, tens of
    # mm lower. On 0003 L1 the superior surface reads (posterior->anterior)
    #     hgt 12 14 14 15 16 17 18 19 23 21  4  2
    # and those last columns are the anterior wall. Because the rim refit deliberately
    # samples the OUTER quartiles, it sampled exactly that contamination and RANSAC
    # locked onto a confident, self-consistent, wrong plane: L1's superior plate came out
    # at -16.6 deg against +20.1 for its own inferior, with rms a healthy 0.74 -- which is
    # why a residual threshold cannot catch this. rms measures how well the points fit the
    # plane that was chosen, not whether the right points were chosen.
    # Keep only surface points NEAR THE PLATE. Past the plate's edge the topmost voxel
    # in a column belongs to the cortical wall, and a mid-body anterior bulge or spur
    # creates columns whose topmost voxel is far below the plate entirely. Measured on
    # 0003, the superior surface carried points 21-26 mm off its own rim plane while the
    # inferior sat at 0.6-1.2 mm -- a real endplate concavity is 1-3 mm, so those were
    # wall, not plate. That contamination tilts the tangent, and because each corner is
    # the tangent x wall intersection, a tilted tangent SLIDES THE CORNER ALONG THE WALL:
    # the superior posterior corner ends up too far anterior.
    #
    # A fixed A-P fraction (the previous edge_trim_frac) cannot express this -- how far
    # in the contamination reaches depends on the bulge, not on a share of the width.
    # Distance from the plate does, and it is what "endplate" means.
    # Distance PERPENDICULAR TO THE PLATE, from the initial robust plane -- not vertical
    # distance from the highest voxel. An endplate is tilted (18 deg is ordinary), and
    # over a 44 mm body that is a 14 mm drop, so a vertical band anchored on the maximum
    # keeps only the sliver near the plate's high end: on 0003 L1 it retained 6.3 mm of a
    # 43.8 mm body, and the tangent was then fitted to that fragment. Perpendicular
    # distance is tilt-invariant, which is what "near the plate" has to mean.
    near_plate = np.abs((surf - c) @ n) <= plate_band_mm
    core = surf[near_plate]
    if len(core) < 8:                        # fall back to the old A-P trim
        t_all0 = surf @ ap
        span0 = float(np.ptp(t_all0))
        core = surf[(t_all0 >= float(np.min(t_all0)) + edge_trim_frac * span0)
                    & (t_all0 <= float(np.max(t_all0)) - edge_trim_frac * span0)]
    if len(core) < 8:
        core = surf
    t_all = surf @ ap
    t_core = core @ ap
    lo_t, hi_t = np.quantile(t_core, [0.25, 0.75])
    rim = core[(t_core <= lo_t) | (t_core >= hi_t)]
    fit2 = fit_plane_ransac(rim, thresh_mm=ransac_mm) if len(rim) >= 6 else None
    if fit2 is not None:
        c, n, _ = fit2
    d_plane = np.abs((surf - c) @ n)
    on_plate = surf[d_plane <= max(ransac_mm, 1.5)]
    if len(on_plate) < 4:
        on_plate = keep
    rms = float(np.sqrt(np.mean(((on_plate - c) @ n) ** 2)))

    # SHAPE MODEL over the near-plate surface. A robust quadratic profile admits exactly
    # one concavity -- the normal biconcave plate -- and cannot represent an osteophyte
    # lip or a Schmorl's node, so Tukey's biweight drives both to zero influence and the
    # tangent stops depending on which voxel happens to be extreme. Unlike the RANSAC
    # plane this keeps every inlier, weighted, instead of committing to one consensus set
    # (which is how a wall-contaminated subset could be fitted confidently, with a small
    # residual, and still be wrong). See fit_profile_robust for the references.
    prof = fit_profile_robust(core @ ap, core @ a_ax, degree=profile_degree)
    if prof is not None:
        pc, pw = prof
        if int((pw > 0).sum()) >= 8:
            xs = (core @ ap)[pw > 0]
            zs = np.vander(xs, profile_degree + 1) @ pc
            # re-anchor the plate on the MODEL: centroid and tangent from the fitted
            # profile rather than from the raw point set
            lr_med = core.mean(axis=0)
            lr_med = lr_med - (lr_med @ ap) * ap - (lr_med @ a_ax) * a_ax
            xa, xb = float(xs.min()), float(xs.max())
            za = float((np.vander([xa], profile_degree + 1) @ pc)[0])
            zb = float((np.vander([xb], profile_degree + 1) @ pc)[0])
            pa = lr_med + xa * ap + za * a_ax
            pb = lr_med + xb * ap + zb * a_ax
            u_prof = pb - pa
            if np.linalg.norm(u_prof) > 1e-6:
                c = 0.5 * (pa + pb)
                n = unit(np.cross(unit(np.cross(u_prof, lrv)), lrv))                     if np.linalg.norm(np.cross(u_prof, lrv)) > 1e-9 else n
                n = unit(np.cross(unit(lr), unit(u_prof)))
                resid = zs - (core @ a_ax)[pw > 0]
                rms = float(np.sqrt(np.mean(resid ** 2)))
                # RE-SELECT the on-plate points against the NEW plane. Leaving them from
                # the previous fit left the corner search anchored on one plane while
                # measuring distances against another, and the bounded wall intersection
                # then clamped to a baseline that no longer meant anything: 0003 L1
                # superior came out 21 mm short posteriorly and L2 10.2 mm.
                d_new = np.abs((surf - c) @ n)
                sel_new = d_new <= max(ransac_mm, 1.5)
                if int(sel_new.sum()) >= 8:
                    on_plate = surf[sel_new]

    # cortical walls, from the body's own A-P margins per height bin
    hgt = pts @ a_ax
    apc = pts @ ap
    nb = 24
    hb = np.floor((hgt - hgt.min()) / (np.ptp(hgt) + 1e-9) * nb).astype(int)
    ant_pts, post_pts = [], []
    for k in np.unique(hb):
        m_k = hb == k
        ant_pts.append(pts[m_k][int(np.argmax(apc[m_k]))])
        post_pts.append(pts[m_k][int(np.argmin(apc[m_k]))])
    ant_pts, post_pts = np.array(ant_pts), np.array(post_pts)
    # drop the top/bottom bins: there the "wall" is really the endplate rim turning over
    def _mid(w):
        if len(w) < 8:
            return w
        h = w @ a_ax
        lo_h, hi_h = np.quantile(h, [0.25, 0.75])
        t = w[(h >= lo_h) & (h <= hi_h)]
        return t if len(t) >= 4 else w
    wall_a = fit_plane_ransac(_mid(ant_pts), thresh_mm=ransac_mm)
    wall_p = fit_plane_ransac(_mid(post_pts), thresh_mm=ransac_mm)

    # the endplate TANGENT, in the sagittal plane, lying in the plate
    u = ap - (ap @ n) * n
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return None
    u = u / nu

    def _hit(wall, sign, max_extend_mm=8.0):
        """Where the tangent meets the wall, BOUNDED by the on-plate extreme.

        The extreme is already within a few mm of the wall -- the intersection only has
        to carry the corner the last bit, over the turn where the cortex curves away.
        So it is accepted only if it lies outward of the extreme by at most
        `max_extend_mm`, and never inward. Unbounded, a wall plane that happens to run
        near-parallel to the tangent throws the corner tens of mm away: measured
        excursions of 30 mm at 0003 L1 and 28 mm past the canal at S1. `sign` is +1 for
        the anterior corner, -1 for the posterior.
        """
        t2 = on_plate @ ap
        base = on_plate[int(np.argmax(t2))] if sign > 0 else on_plate[int(np.argmin(t2))]
        base = base - ((base - c) @ n) * n
        t_base = float((base - c) @ u)
        if wall is not None:
            cw, nw, _ = wall
            den = float(u @ nw)
            if abs(den) > 1e-6:
                t_w = float((cw - c) @ nw) / den
                # outward of the extreme, by no more than max_extend_mm
                lo, hi = (t_base, t_base + max_extend_mm) if sign > 0 else                          (t_base - max_extend_mm, t_base)
                if lo <= t_w <= hi:
                    return c + t_w * u
        return base

    return _hit(wall_a, +1), _hit(wall_p, -1), rms, int(len(on_plate))


def body_mask(mask, affine, *, sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
              core_mm: float = 4.0, canal_from=None) -> Optional[np.ndarray]:
    """The vertebral BODY: canal criterion, falling back to the watershed.

    Canal first because it is level-independent; watershed only when the arch does not
    enclose a canal (incomplete or lytic arch), where a thin-neck cut is the best
    available. None if neither succeeds -- callers should drop the level rather than
    measure a body that was never isolated.
    """
    b = body_mask_by_canal(mask, affine, sup_axis=sup_axis, lr=lr, canal_from=canal_from)
    if b is not None and b.any():
        return b
    return body_mask_watershed(mask, affine, core_mm=core_mm)


# ---------------------------------------------------------------------------
# Frobin quadrilateral: corners as endplate x wall intersections
# ---------------------------------------------------------------------------

def _fit_line_ransac(xy, *, thresh_mm: float = 1.5, iters: int = 300, seed: int = 0):
    """Robust 2-D line as (point, unit direction), or None. RANSAC, same rationale as
    the plane fit: an osteophyte is a gross outlier and must not tilt the line."""
    P = np.asarray(xy, float)
    if len(P) < 2:
        return None
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iters)):
        i, j = rng.choice(len(P), size=2, replace=False)
        d = P[j] - P[i]
        nn = np.linalg.norm(d)
        if nn < 1e-9:
            continue
        d = d / nn
        nrm = np.array([-d[1], d[0]])
        inl = np.abs((P - P[i]) @ nrm) <= thresh_mm
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 2:
        return None
    Q = P[best]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    return c, vt[0] / np.linalg.norm(vt[0])


def _intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    det = np.linalg.det(A)
    if abs(det) < 1e-9:                      # parallel: no corner
        return None
    t = np.linalg.solve(A, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + t[0] * np.asarray(d1, float)


# !! NOT WIRED IN, AND CURRENTLY BROKEN. Kept because the FORMULATION is right -- a
# !! parametric quadrilateral fitted to the body is the cleanest statement of what a
# !! corner is -- but the fit is not. Measured on 0003 the superior plate margins fit at
# !! rms 6-10 mm against 0.3-1.7 for the inferior ones, and several corners come back
# !! NaN. Cause is known, not yet solved: past the plate's edge the extreme voxel in a
# !! column belongs to the cortical WALL, and Tukey's IRLS starts from an UNWEIGHTED
# !! least-squares fit, so a large contaminated set inflates the initial residual scale
# !! and then survives its own rejection test. Trimming the outer columns did not fix it.
# !! What it needs is a robust SEED (RANSAC or repeated-median) before the IRLS so the
# !! initial scale is not set by the contamination. The LIVE path is
# !! endplate_corners_body + xrsp.oblique.extend_in_body.
def body_quad_corners(mask, affine, *, sup_axis=WORLD_SUPERIOR, lr=(1.0, 0.0, 0.0),
                      lat_frac: float = 0.45, ransac_mm: float = 1.5,
                      edge_frac: float = 0.30, canal_from=None, body=None,
                      min_points: int = 30):
    """All four vertebral body corners, as endplate x wall INTERSECTIONS.

    The Frobin method: in the midsagittal plane the vertebral body is a quadrilateral
    bounded by the superior and inferior endplates and the anterior and posterior
    cortical walls, and the corners are where those four lines meet.

      Frobin W, Brinckmann P, Biggemann M, Tillotson M, Burton K. "Precision
        measurement of disc height, vertebral height and sagittal plane displacement
        from lateral radiographic views of the lumbar spine."
        Clin Biomech 1997;12(Suppl 1):S1-S63.
      Hurxthal LM. "Measurement of anterior vertebral compressions and biconcave
        vertebrae." Am J Roentgenol 1968;103(3):635-644.
      Black DM, Palermo L, Nevitt MC, et al. "Comparison of methods for defining
        prevalent vertebral deformities." J Bone Miner Res 1995;10(6):890-902.

    Why an intersection rather than the extreme surface voxel: a corner is a CONSTRUCTED
    point, not an observed one. Taking the last voxel on the endplate stops wherever the
    cortex begins to curve up into the posterior wall, which lands the posterosuperior
    corner short -- visible on an overlay as a posterior corner that quits before the
    body surface turns. Extending the endplate tangent to meet the wall line is exactly
    what a reader draws, and it stays well defined where the surface itself is not.

    Osteophyte handling comes free: each of the four lines is fitted by RANSAC over its
    own cortical margin, and a spondylophyte is a gross outlier to both the plate it
    grows from and the wall it projects past, so it cannot move either line. This is the
    morphometric convention -- measure the body, ignore the spur.

    Returns dict with sup_ant / sup_post / inf_ant / inf_post as world-mm points, plus
    per-line rms and inlier counts, or None.
    """
    m = np.asarray(mask, bool)
    if body is None:
        body = body_mask(m, affine, sup_axis=sup_axis, lr=lr, canal_from=canal_from)
    if body is None or body.sum() < min_points:
        return None
    idx = np.array(np.nonzero(body)).T
    P = (np.c_[idx, np.ones(len(idx))] @ np.asarray(affine, float).T)[:, :3]
    a = unit(sup_axis)
    lrv = unit(lr)
    ap = anterior_axis(a, lr)
    if 0.0 < lat_frac < 1.0:                 # midsagittal band: drops TPs and sacral alae
        lp = P @ lrv
        lo, hi = np.quantile(lp, [(1 - lat_frac) / 2, 1 - (1 - lat_frac) / 2])
        P = P[(lp >= lo) & (lp <= hi)]
    if len(P) < min_points:
        return None
    mid_lr = float(np.median(P @ lrv))
    x, y = P @ ap, P @ a                     # 2-D midsagittal: anterior, cranial

    def margin(u, v, take_max, bins=24):
        """Extreme v per bin of u -- one cortical margin of the outline."""
        b = np.floor((u - u.min()) / (np.ptp(u) + 1e-9) * bins).astype(int)
        out = []
        for k in np.unique(b):
            s = b == k
            j = np.argmax(v[s]) if take_max else np.argmin(v[s])
            out.append([u[s][j], v[s][j]])
        return np.array(out)

    # FOUR MARGINS of the midsagittal outline, each fitted with the SAME robust
    # degree-2 profile model (fit_profile_robust): two endplates and two cortical walls.
    # One estimator, one code path, all in 3-D -- the corners are then the four
    # intersections, i.e. a parametric quadrilateral fitted to the body while osteophytes
    # and Schmorl's nodes carry zero weight. Degree 2 gives the endplates their single
    # normal concavity and the walls their waisting, and cannot represent a spur.
    sup_m = margin(x, y, True)               # superior endplate:  y = f(x)
    inf_m = margin(x, y, False)              # inferior endplate:  y = f(x)
    ant_m = margin(y, x, True)               # anterior wall:      x = g(y)
    post_m = margin(y, x, False)             # posterior wall:     x = g(y)

    def _prof(mm, deg=2, trim=0.0):
        """Robust profile, optionally trimming the outermost columns first.

        The endplate margins need the trim. Past the plate's edge the extreme voxel in a
        column is the cortical WALL, tens of mm away, and Tukey's IRLS starts from an
        UNWEIGHTED least-squares fit -- so a large contaminated set inflates the initial
        residual scale and then survives its own rejection test. Measured: superior
        margins fitted at rms 3.3-8.2 mm against 0.3-1.3 for the inferior ones, the same
        asymmetry that broke the other path. The corners come from intersections, so
        dropping those columns costs nothing.
        """
        if len(mm) < 6:
            return None
        m2 = mm
        if trim > 0.0 and len(mm) >= 10:
            span = float(np.ptp(mm[:, 0]))
            lo = float(np.min(mm[:, 0])) + trim * span
            hi = float(np.max(mm[:, 0])) - trim * span
            k = mm[(mm[:, 0] >= lo) & (mm[:, 0] <= hi)]
            if len(k) >= 6:
                m2 = k
        out = fit_profile_robust(m2[:, 0], m2[:, 1], degree=deg, min_points=5)
        return None if out is None else out[0]

    Pf = {"sup": _prof(sup_m, trim=edge_frac), "inf": _prof(inf_m, trim=edge_frac),
          "ant": _prof(ant_m, trim=edge_frac), "post": _prof(post_m, trim=edge_frac)}
    if any(v is None for v in Pf.values()):
        return None

    def _corner(plate, wall):
        """Where an endplate profile y=f(x) meets a wall profile x=g(y).

        Solved by a few fixed-point steps from the wall's value at the plate's mid
        height: both curves are gentle, so this converges in a handful of iterations and
        avoids solving a quartic. Deterministic -- no random component.
        """
        y0 = float(np.median(sup_m[:, 1] if plate == "sup" else inf_m[:, 1]))
        xg = float(np.polyval(Pf[wall], y0))
        for _ in range(24):
            yf = float(np.polyval(Pf[plate], xg))
            xn = float(np.polyval(Pf[wall], yf))
            if abs(xn - xg) < 1e-4:
                xg = xn
                break
            xg = 0.5 * (xg + xn)             # damped, so a steep wall cannot oscillate
        return np.array([xg, float(np.polyval(Pf[plate], xg))])

    def to_world(p2):
        """Midsagittal (anterior, cranial) back to world mm."""
        return mid_lr * lrv + float(p2[0]) * ap + float(p2[1]) * a

    # clamp into the observed outline: a polynomial extrapolates fast and a corner
    # cannot lie outside the bone
    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())

    out = {}
    for name, (pl, wl) in (("sup_ant", ("sup", "ant")), ("sup_post", ("sup", "post")),
                           ("inf_ant", ("inf", "ant")), ("inf_post", ("inf", "post"))):
        q = _corner(pl, wl)
        q[0] = float(np.clip(q[0], x_lo, x_hi))
        q[1] = float(np.clip(q[1], y_lo, y_hi))
        out[name] = to_world(q)

    def rms_of(k, mm, swap=False):
        c = Pf[k]
        return float(np.sqrt(np.mean((mm[:, 1] - np.polyval(c, mm[:, 0])) ** 2)))
    out["rms"] = {"sup": rms_of("sup", sup_m), "inf": rms_of("inf", inf_m),
                  "ant": rms_of("ant", ant_m), "post": rms_of("post", post_m)}
    return out


# ---------------------------------------------------------------------------
# Robust parametric endplate profile (shape model)
# ---------------------------------------------------------------------------

def fit_profile_robust(x, z, *, degree: int = 2, tukey_c: float = 4.685,
                       max_iter: int = 12, min_points: int = 8):
    """Robust low-order polynomial z(x) by IRLS with Tukey's biweight.

    A SHAPE MODEL for the endplate profile, in the spirit of the parametric vertebral
    body models used to make morphometry robust to pathology:
      Stern D, Likar B, Pernus F, Vrtovec T. "Parametric modelling and segmentation of
        vertebral bodies in 3D CT and MR spine images." Phys Med Biol 2011;56(23):7505-22.
      Roberts M, Cootes TF, Adams JE. "Vertebral morphometry: semiautomatic determination
        of detailed shape from DXA images using active appearance models."
        Invest Radiol 2006;41(12):849-859.
      de Bruijne M, Lund MT, Tanko LB, Pettersen PC, Nielsen M. "Quantitative vertebral
        morphometry using neighbor-conditional shape models."
        Med Image Anal 2007;11(5):503-512.

    Why a model rather than the voxels: degree 2 admits exactly ONE concavity, which is
    the normal biconcave endplate (Hurxthal's middle height, Am J Roentgenol 1968, exists
    for that reason). It cannot represent a localised spur or a Schmorl's node divot, so
    both show up as large residuals and Tukey's biweight gives them zero weight -- they
    are excluded because they do not fit the anatomy, not because a threshold was tuned
    to them. Raising `degree` would let the model absorb the pathology and defeat this.

    Unlike RANSAC this keeps ALL of the plate: every inlier contributes, weighted. RANSAC
    picks one consensus set and discards the rest, which is why it could lock confidently
    onto a wall-contaminated subset and still report a small residual.

    Returns (coeffs_highest_first, weights) or None.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    if len(x) < max(min_points, degree + 1):
        return None
    V = np.vander(x, degree + 1)
    w = np.ones(len(x))
    coef = None
    for _ in range(int(max_iter)):
        W = np.sqrt(w)[:, None]
        try:
            coef, *_ = np.linalg.lstsq(V * W, z * np.sqrt(w), rcond=None)
        except np.linalg.LinAlgError:
            return None
        r = z - V @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9   # robust sigma (MAD)
        u = r / (tukey_c * s)
        w_new = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if np.allclose(w_new, w, atol=1e-3):
            w = w_new
            break
        w = w_new
        if w.sum() < max(min_points, degree + 1):
            return None
    return (coef, w) if coef is not None else None


def endplate_chord_from_profile(surf, ap, sup_axis, x_ant, x_post, *, degree: int = 2):
    """The clinical endplate CHORD, taken from a robust profile model.

    The plate is modelled as z(x) over the A-P coordinate and the chord is drawn between
    the model's values at the anterior and posterior wall positions -- the corner-to-
    corner tangent a reader draws, but with the endpoints coming from a fitted shape
    rather than from whichever voxel happens to be extreme. An osteophyte at the rim or a
    Schmorl's node mid-plate therefore cannot tilt the line.

    Returns (anterior_point, posterior_point, rms_of_inliers, n_inliers) or None.
    """
    S = np.asarray(surf, float)
    a_ax = unit(sup_axis)
    apv = unit(ap)
    x = S @ apv
    z = S @ a_ax
    fit = fit_profile_robust(x, z, degree=degree)
    if fit is None:
        return None
    coef, w = fit
    inl = w > 0.0
    if int(inl.sum()) < 8:
        return None
    resid = z[inl] - np.vander(x[inl], degree + 1) @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # in-plane component perpendicular to both ap and the plate's L-R spread
    def _pt(xv):
        zv = float((np.vander([float(xv)], degree + 1) @ coef)[0])
        # rebuild in world: keep the surface's median L-R, move along ap and sup
        lr_med = S.mean(axis=0) - (S.mean(axis=0) @ apv) * apv - (S.mean(axis=0) @ a_ax) * a_ax
        return lr_med + float(xv) * apv + zv * a_ax
    return _pt(x_ant), _pt(x_post), rms, int(inl.sum())
