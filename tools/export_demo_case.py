"""export_demo_case.py — turn one CTSpinoPelvic1K case into a web demo bundle.

Produces, under <out-dir>/<case-id>/ :
    ct.nii.gz      cropped (and optionally bone-masked / downsampled) CT
    seg.nii.gz     matching label map
    metrics.json   ostk spinopelvic summary + world-mm drawing geometry
and appends the case to <out-dir>/manifest.json.

The CT is the size problem (a full spinopelvic CT is 100-300 MB). We:
  * crop to the label bounding box + margin (the spinopelvic FOV a reader needs),
  * optionally zero everything outside the dilated bone (`--mask-bone`) so gzip
    shrinks the air/soft-tissue to almost nothing,
  * optionally subsample (`--downsample N`).
World coordinates are preserved through cropping/downsampling (the affine is
adjusted), so the mm landmarks ostk computes still line up with the volume.

Usage:
    python tools/export_demo_case.py \
        --ct 0002_ct.nii.gz --label 0002_label.nii.gz \
        --case-id 0002 --out-dir ../openspineconsortium.github.io/pacs/data \
        --crop-margin 25 --mask-bone --downsample 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ostk import geometry as g       # noqa: E402
from ostk import metrics             # noqa: E402
from ostk import spine               # noqa: E402
from ostk.io import load_ct, load_label, voxels_to_world  # noqa: E402
from ostk.labels import lid          # noqa: E402
from ostk.masks import (binary_mask, endplate_points, largest_component,  # noqa: E402
                        mask_world, surface_slab)

WORLD_SUP = g.WORLD_SUPERIOR
RAY = 130.0          # default annotation ray length (mm)


# --------------------------------------------------------------------------- #
# drawing geometry (world mm) — everything projected into ONE sagittal plane   #
# so the whole construction is visible on a single sagittal slice.             #
# --------------------------------------------------------------------------- #

def _femoral_axis(label, affine, head_frac=0.35, min_voxels=30):
    cs = []
    for fem in ("femur_left", "femur_right"):
        w = mask_world(largest_component(binary_mask(label, lid(fem))), affine)
        head = surface_slab(w, WORLD_SUP, "superior", head_frac)
        if len(head) < min_voxels:
            return None
        c, _, _ = g.fit_sphere(head)
        cs.append(c)
    return np.asarray(cs[0]), np.asarray(cs[1])     # left, right


def _endplate(label, affine, level, neighbor=None, min_voxels=30):
    """Superior-endplate (centroid, cranial unit normal, rms) via the shared
    `ostk.spine` primitive (anterior-body + true-surface fit). `neighbor` is no
    longer needed but kept for call-site compatibility."""
    return spine.endplate_from_label(label, affine, level, which="superior",
                                     min_points=min_voxels)


def _s1_endplate_surface(label, affine):
    """The cleaned S1 superior-endplate surface points (world mm), so the drawn
    endplate line spans the true endplate and PI/SS/PT anchor on its geometric
    centre. Returns the (N,3) surface or None."""
    src = "S1" if binary_mask(label, lid("S1")).any() else "sacrum"
    pts = mask_world(largest_component(binary_mask(label, lid(src))), affine)
    res = spine.endplate_corners(pts, which="superior",
                                 **spine.corner_params_for_level("S1"))
    return None if res is None else np.asarray(res[2], float)


def _endplate_corners(label, affine, level):
    """(anterior_corner, posterior_corner) world mm of a superior endplate, so the
    drawn endplate line covers the actual endplate. Returns None if unavailable."""
    src = level
    if level == "S1" and not binary_mask(label, lid("S1")).any():
        src = "sacrum"
    pts = mask_world(largest_component(binary_mask(label, lid(src))), affine)
    res = spine.endplate_corners(pts, which="superior",
                                 **spine.corner_params_for_level(level))
    return None if res is None else (np.asarray(res[0], float), np.asarray(res[1], float))


def _project(p, origin, lr):
    p = np.asarray(p, float)
    return p - ((p - origin) @ lr) * lr


def _p(v):
    return [round(float(x), 2) for x in v]


def _seg(p, q):
    return [_p(p), _p(q)]


def _intersect(p0, d0, p1, d1):
    """Intersection of two COPLANAR 3-D lines (p0+t*d0, p1+s*d1) via least squares."""
    A = np.column_stack([np.asarray(d0, float), -np.asarray(d1, float)])  # 3x2
    b = np.asarray(p1, float) - np.asarray(p0, float)
    ts, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.asarray(p0, float) + ts[0] * np.asarray(d0, float)


def _angle_entry(name, label, value, color, solid, dashed, arc, label_at, rule=None,
                 arc_r_px=None):
    """solid: [p,q] mm pairs drawn SOLID (the anatomical endplate line). dashed:
    [p,q] pairs drawn DOTTED (reference/construction lines — HRL, VRL, perpendicular,
    pelvic radius). arc: {center,a,b} mm angle wedge. label_at: mm point for the text.
    rule (optional): {dots:[mm,...], marks:[{pos,text},...]} — endpoint/midpoint dots
    and half-length callouts on the endplate line."""
    d = {"id": name, "label": label,
         "value": None if value is None else round(float(value), 1), "units": "°",
         "color": color, "segments": solid, "dashed": dashed,
         "arc": {"center": _p(arc[0]), "a": _p(arc[1]), "b": _p(arc[2])},
         "label_at": _p(label_at)}
    if rule is not None:
        d["rule"] = rule
    if arc_r_px is not None:
        d["arc_r_px"] = arc_r_px
    return d


def build_geometry(label, affine):
    """Assemble the angle annotations (world mm) for whatever is computable."""
    fem = _femoral_axis(label, affine)
    s1 = _endplate(label, affine, "S1", neighbor="L5")   # S1 endplate faces L5
    l1 = _endplate(label, affine, "L1", neighbor="T12")  # L1 endplate faces T12 (if in FOV)

    # sagittal plane: normal = L-R axis (bicoxofemoral if femurs present, else image X)
    if fem is not None:
        cL, cR = fem
        lr = g.unit(cR - cL)
    else:
        lr = np.array([1.0, 0.0, 0.0])
    # plane passes through whichever anchor we have
    origin = s1[0] if s1 is not None else (l1[0] if l1 is not None else np.zeros(3))
    sup_s = g.unit(g.project_out(WORLD_SUP, lr))           # vertical in plane
    horiz = g.unit(np.cross(lr, sup_s))                    # horizontal in plane

    # medial viewing slice = L-R centre of the spinal COLUMN (vertebrae + sacrum),
    # so the demo opens on the spine midline — not the S1-endplate plane, which on a
    # rotated/curved spine is laterally offset from the lumbar bodies.
    col_ids = []
    for nm in ("sacrum", "S1", "L1", "L2", "L3", "L4", "L5", "L6",
               "T13", "T12", "T11", "T10", "T9", "T8"):
        try:
            col_ids.append(lid(nm))
        except Exception:
            pass
    colmask = np.isin(label, col_ids)
    view_center = (np.median(mask_world(colmask, affine), axis=0)
                   if colmask.any() else origin)

    angles, points = [], []
    M = _project(0.5 * (cL + cR), origin, lr) if fem is not None else None
    if fem is not None:
        points += [{"id": "bicoxofemoral", "pos": _p(M)}]

    if s1 is not None and fem is not None:
        # The S1 endplate line is the ACTUAL traced corner-to-corner segment, and the
        # construction anchors P on ITS midpoint (the Legaye/Greenberg "1/2 1/2"
        # point) — so PI/SS/PT point to the true sacral-endplate body midpoint, not a
        # fixed-width stub's centre.
        n_s = g.unit(g.project_out(s1[1], lr))
        if n_s @ sup_s < 0:
            n_s = -n_s
        e_dir = g.unit(np.cross(lr, n_s))                  # S1 endplate line direction
        # P = over-mask endplate midpoint, ON the rim (== ostk.metrics' PI/PT radius
        # origin, so the drawn angles match the report). Draw the endplate line CENTRED
        # on P, spanning the over-mask portion. (S1 only; the LL/L1 line is untouched.)
        om = spine.endplate_overmask_midpoint_from_label(label, affine, "S1")
        P = _project(om, origin, lr) if om is not None else _project(s1[0], origin, lr)
        surf = _s1_endplate_surface(label, affine)
        if surf is not None and len(surf) >= 6:
            surf_p = surf - ((surf - origin) @ lr)[:, None] * lr   # project to sag plane
            proj = (surf_p - P) @ e_dir
            half = 0.5 * float(np.percentile(proj, 97.0) - np.percentile(proj, 3.0))
        else:
            half = 26.0
        end_a, end_b = P - half * e_dir, P + half * e_dir
        s1line = _seg(end_a, end_b)
        # Legaye "1/2 + 1/2" rule with REAL measurements: ONE dot at the midpoint,
        # dotted perpendicular ticks at the two ends + midpoint, and a <-> arrow over
        # each half (above the line, along the endplate normal) with its length.
        TL, AO = 20.0, 14.0                            # tick length, arrow offset (mm)
        ss_rule = {
            "mid": _p(P),
            "ticks": [_seg(end_a, end_a + TL * n_s), _seg(P, P + TL * n_s),
                      _seg(end_b, end_b + TL * n_s)],
            "spans": [
                {"a": _p(end_a + AO * n_s), "b": _p(P + AO * n_s),
                 "label": _p(0.5 * (end_a + P) + (AO + 8) * n_s), "text": f"{half:.1f} mm"},
                {"a": _p(P + AO * n_s), "b": _p(end_b + AO * n_s),
                 "label": _p(0.5 * (P + end_b) + (AO + 8) * n_s), "text": f"{half:.1f} mm"},
            ],
        }
        radius = g.unit(P - M)                             # hip-axis -> S1 midpoint
        PI = g.angle_between(n_s, radius)
        SS = g.angle_between(e_dir, horiz)
        PT = g.angle_between(radius, sup_s)
        # in-plane anterior/posterior so the HRL projects POSTERIOR like the figure
        ant_p = g.unit(g.project_out(np.array([0.0, 1.0, 0.0]), lr))
        if ant_p[1] < 0:
            ant_p = -ant_p
        horiz_post = horiz if horiz @ ant_p < 0 else -horiz
        horiz_ant = -horiz_post                            # anterior horizontal (dynamic)
        e_post = e_dir if e_dir @ ant_p < 0 else -e_dir
        HRLL, PERP, VRLL = 92.0, 80.0, 92.0
        points += [{"id": "s1_midpoint", "pos": _p(P)}]    # the "1/2 1/2" anchor
        # SS: S1 endplate vs HRL — the HRL ORIGINATES at the midpoint and projects
        # posterior; label written along the HRL.
        # dotted continuation of the endplate posteriorly + the HRL, with the SS arc
        # drawn out at the end region (Legaye fig.): angle between the continued
        # endplate and the horizontal.
        angles.append(_angle_entry(
            "SS", "Sacral Slope", SS, "#60a5fa",
            [s1line],
            [_seg(P + half * e_post, P + (half + 70.0) * e_post),
             _seg(P, P + HRLL * horiz_post)],
            (P, P + 44 * e_post, P + 44 * horiz_post),
            P + 78 * horiz_post + 16 * sup_s, rule=ss_rule, arc_r_px=92))
        # PI: S1-endplate perpendicular (into the pelvis) vs the pelvic radius to the
        # femoral-head axis; wedge at the S1 midpoint. Label sits slightly POSTERIOR
        # (dynamic) so it doesn't collide with PT.
        angles.append(_angle_entry(
            "PI", "Pelvic Incidence", PI, "#36d399",
            [s1line], [_seg(P, P - PERP * n_s), _seg(P, M)],
            (P, P - 46 * n_s, P + 46 * g.unit(M - P)),
            P + g.unit(g.unit(M - P) - n_s) * 44))   # in the arc (bisector of PI)
        # PT: pelvic radius vs vertical (VRL), wedge at the femoral-head axis. Label
        # on the ANTERIOR side (dynamic) so PI and PT can be read at the same time.
        angles.append(_angle_entry(
            "PT", "Pelvic Tilt", PT, "#fbbf24",
            [], [_seg(M - 16 * sup_s, M + VRLL * sup_s), _seg(M, P)],
            (M, M + 46 * sup_s, M + 46 * radius),
            M + 14 * horiz_ant + 52 * sup_s))        # hugging the VRL, anterior side

    if s1 is not None and l1 is not None:
        P1, n1, _ = l1
        P7, n7, _ = s1
        P1 = _project(P1, origin, lr)
        P7 = _project(P7, origin, lr)
        n1s = g.unit(g.project_out(n1, lr)); n1s = n1s if n1s @ sup_s >= 0 else -n1s
        n7s = g.unit(g.project_out(n7, lr)); n7s = n7s if n7s @ sup_s >= 0 else -n7s
        e1 = g.unit(np.cross(lr, n1s))
        e7 = g.unit(np.cross(lr, n7s))
        LL = g.cobb_angle(n1, n7, lr)
        HW = 34.0
        # Cobb construction (Greenberg Fig. 73.1), FULLY precomputed in world mm so
        # the viewer only maps fixed points (no screen-space re-derivation -> can't
        # flip on scroll). The endplate lines COVER the endplate (corner to corner)
        # and extend only on the ANGLE side (anterior) to the perpendicular; the
        # perpendiculars meet at X, L1's STOPS at X and S1's continues past it.
        ant = g.unit(np.cross(lr, sup_s))                  # anterior in-plane axis
        if ant @ np.array([0.0, 1.0, 0.0]) < 0:
            ant = -ant
        e1a = e1 if e1 @ ant >= 0 else -e1                 # endplate dirs -> anterior
        e7a = e7 if e7 @ ant >= 0 else -e7
        c1, c7 = _endplate_corners(label, affine, "L1"), _endplate_corners(label, affine, "S1")
        if c1 is not None and c7 is not None:
            Ac1, Pc1 = _project(c1[0], origin, lr), _project(c1[1], origin, lr)
            Ac7, Pc7 = _project(c7[0], origin, lr), _project(c7[1], origin, lr)
            A0, A1 = Ac1, Ac7                              # perpendiculars erected at the corner
            # SOLID endplate line covers the endplate exactly (terminates at the
            # corners); only the perpendicular below extends past it.
            l1_line, s1_line = _seg(Pc1, Ac1), _seg(Pc7, Ac7)
        else:
            A0, A1 = P1 + HW * e1a, P7 + HW * e7a
            l1_line, s1_line = _seg(P1 - HW * e1, P1 + HW * e1), _seg(P7 - HW * e7, P7 + HW * e7)
        X = _intersect(A0, n1s, A1, n7s)                   # perpendiculars meet here
        beyond1 = X + (X - A1) * 0.75                      # S1 perpendicular past X
        bis = g.unit(g.unit(A0 - X) + g.unit(beyond1 - X))
        angles.append(_angle_entry(
            "LL", "Lumbar Lordosis", LL, "#f472b6",
            [l1_line, s1_line],
            [_seg(A0, X), _seg(A1, beyond1)],
            (X, A0, beyond1), X + bis * 54))

    return {"sagittal_normal": [round(float(x), 4) for x in lr],
            "plane_origin": _p(origin), "view_center": _p(view_center),
            "angles": angles, "points": points}


# --------------------------------------------------------------------------- #
# volume cropping / masking / downsampling                                     #
# --------------------------------------------------------------------------- #

def _bbox(mask, margin_vox):
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - margin_vox, 0)
    hi = np.minimum(idx.max(0) + margin_vox + 1, mask.shape)
    return lo, hi


def _crop_affine(affine, lo, step):
    a = affine.copy().astype(float)
    a[:3, 3] = (affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]
    a[:3, :3] = affine[:3, :3] * step
    return a


def process(args):
    import nibabel as nib

    label, laff = load_label(args.label)
    seg = label
    # fast path: rebuild only metrics.json (geometry + summary) from the label,
    # reusing the already-written ct/seg bundles. Seconds, vs reloading the raw CT.
    if getattr(args, "geometry_only", False):
        geom = build_geometry(seg, laff)
        summary = metrics.spinopelvic_summary_from_label(seg, laff, case_id=args.case_id)
        ll = next((a for a in geom["angles"] if a["id"] == "LL" and a["value"] is not None), None)
        if ll:
            summary["LL"] = ll["value"]
        mpath = os.path.join(args.out_dir, args.case_id, "metrics.json")
        meta = json.load(open(mpath, encoding="utf-8"))
        meta["summary"], meta["geometry"] = summary, geom
        if args.title:
            meta["label"] = args.title
        json.dump(meta, open(mpath, "w", encoding="utf-8"), indent=2, default=_jdef)
        print(f"[{args.case_id}] geometry-only -> "
              f"{[a['id'] + '=' + str(a['value']) for a in geom['angles']]}")
        return
    ct = caff = None
    if args.ct:
        ct, caff = load_ct(args.ct)
        if ct.shape != seg.shape:
            raise SystemExit("CT and label grids differ; resample first")

    geom = build_geometry(seg, laff)
    summary = metrics.spinopelvic_summary_from_label(seg, laff, case_id=args.case_id)
    # keep the report's LL identical to the drawn construction (neighbour-based)
    ll_ang = next((a for a in geom["angles"] if a["id"] == "LL" and a["value"] is not None), None)
    if ll_ang:
        summary["LL"] = ll_ang["value"]

    fg = (seg > 0) & (seg != 50) & (seg != 255)
    margin_vox = int(round(args.crop_margin / float(abs(laff[0, 0]) or 1.0)))
    lo, hi = _bbox(fg, margin_vox)
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    seg_c = seg[sl]
    ct_c = ct[sl] if ct is not None else None

    if args.mask_bone and ct_c is not None:
        from scipy import ndimage
        keep = ndimage.binary_dilation(seg_c > 0, iterations=args.bone_dilate)
        ct_c = np.where(keep, ct_c, -1000).astype(np.int16)

    step = max(1, int(args.downsample))
    if step > 1:
        seg_c = seg_c[::step, ::step, ::step]
        if ct_c is not None:
            ct_c = ct_c[::step, ::step, ::step]
    out_aff = _crop_affine(laff, lo, step)

    case_dir = os.path.join(args.out_dir, args.case_id)
    os.makedirs(case_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(seg_c.astype(np.int16), out_aff),
             os.path.join(case_dir, "seg.nii.gz"))
    if ct_c is not None:
        nib.save(nib.Nifti1Image(ct_c.astype(np.int16), out_aff),
                 os.path.join(case_dir, "ct.nii.gz"))

    meta = {"case_id": args.case_id, "summary": summary, "geometry": geom,
            "files": {"ct": "ct.nii.gz" if ct_c is not None else None,
                      "seg": "seg.nii.gz"},
            "label": args.title or f"Case {args.case_id}"}
    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=_jdef)

    _update_manifest(args.out_dir, args.case_id, meta["label"])

    def _mb(p):
        return f"{os.path.getsize(p) / 1e6:.1f} MB" if os.path.exists(p) else "-"
    print(f"[{args.case_id}] seg {_mb(os.path.join(case_dir, 'seg.nii.gz'))}  "
          f"ct {_mb(os.path.join(case_dir, 'ct.nii.gz'))}  "
          f"angles: {[a['id'] + ('=' + str(a['value']) if a['value'] is not None else '?') for a in geom['angles']]}")
    if summary["qc_flags"] != ["ok"]:
        print(f"    qc: {summary['qc_flags']}")


def _jdef(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(str(type(o)))


def _update_manifest(out_dir, case_id, label):
    path = os.path.join(out_dir, "manifest.json")
    cases = []
    if os.path.exists(path):
        cases = json.load(open(path, encoding="utf-8")).get("cases", [])
    cases = [c for c in cases if c["id"] != case_id]
    cases.append({"id": case_id, "label": label, "dir": case_id})
    cases.sort(key=lambda c: c["id"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"cases": cases}, fh, indent=2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True)
    p.add_argument("--ct")
    p.add_argument("--case-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--crop-margin", type=float, default=25.0, help="mm around bone bbox")
    p.add_argument("--mask-bone", action="store_true", help="zero non-bone CT (smaller)")
    p.add_argument("--bone-dilate", type=int, default=3, help="voxels to keep around bone")
    p.add_argument("--downsample", type=int, default=1, help="subsample factor")
    p.add_argument("--geometry-only", action="store_true",
                   help="rebuild only metrics.json from the label; reuse ct/seg")
    process(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
