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


def _endplate(label, affine, level, frac=0.15, min_voxels=30):
    src = binary_mask(label, lid(level))
    if level == "S1" and not src.any():
        src = binary_mask(label, lid("sacrum"))
    pts = endplate_points(largest_component(src), affine, WORLD_SUP, "superior", frac)
    if len(pts) < min_voxels:
        return None
    c, n, rms = g.fit_plane_tls(pts)
    if n @ WORLD_SUP < 0:
        n = -n
    return c, n, rms


def _project(p, origin, lr):
    p = np.asarray(p, float)
    return p - ((p - origin) @ lr) * lr


def _p(v):
    return [round(float(x), 2) for x in v]


def _seg(p, q):
    return [_p(p), _p(q)]


def _angle_entry(name, label, value, color, segments, arc, label_at):
    """segments: list of [p,q] mm line pairs to draw (animated). arc: {center,a,b}
    in mm defining the angle wedge. label_at: mm point for the value text."""
    return {"id": name, "label": label,
            "value": None if value is None else round(float(value), 1), "units": "°",
            "color": color, "segments": segments,
            "arc": {"center": _p(arc[0]), "a": _p(arc[1]), "b": _p(arc[2])},
            "label_at": _p(label_at)}


def build_geometry(label, affine):
    """Assemble the angle annotations (world mm) for whatever is computable."""
    fem = _femoral_axis(label, affine)
    s1 = _endplate(label, affine, "S1")
    l1 = _endplate(label, affine, "L1")

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

    angles, points = [], []
    M = _project(0.5 * (cL + cR), origin, lr) if fem is not None else None
    if fem is not None:
        points += [{"id": "bicoxofemoral", "pos": _p(M)}]

    if s1 is not None and fem is not None:
        P = _project(s1[0], origin, lr)
        n_s = g.unit(g.project_out(s1[1], lr))
        if n_s @ sup_s < 0:
            n_s = -n_s
        e_dir = g.unit(np.cross(lr, n_s))                  # S1 endplate line direction
        radius = g.unit(M - P)
        PI = g.angle_between(n_s, radius)
        SS = g.angle_between(e_dir, horiz)
        PT = g.angle_between(radius, sup_s)
        HW = 28.0
        # PI: pelvic radius (P->M) vs S1-endplate perpendicular (P->n_s); wedge at P
        angles.append(_angle_entry(
            "PI", "Pelvic Incidence", PI, "#36d399",
            [_seg(P, M), _seg(P, P + RAY * n_s), _seg(P - HW * e_dir, P + HW * e_dir)],
            (P, M, P + RAY * n_s), P + 0.5 * (g.unit(M - P) + n_s) * 40))
        # SS: S1 endplate line vs horizontal, wedge at P
        angles.append(_angle_entry(
            "SS", "Sacral Slope", SS, "#60a5fa",
            [_seg(P - HW * e_dir, P + HW * e_dir), _seg(P, P + 0.8 * RAY * horiz)],
            (P, P + e_dir, P + horiz), P + 30 * horiz + 14 * sup_s))
        # PT: vertical vs hip-axis->S1 line, wedge at M
        angles.append(_angle_entry(
            "PT", "Pelvic Tilt", PT, "#fbbf24",
            [_seg(M, P), _seg(M, M + RAY * sup_s)],
            (M, P, M + sup_s), M + 0.5 * (g.unit(P - M) + sup_s) * 45))

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
        # Cobb: the L1 and S1 superior-endplate lines + perpendiculars; wedge
        # between the two perpendiculars (== angle between endplates == LL), drawn
        # at the midpoint between the endplates so it sits on the spine.
        mid = 0.5 * (P1 + P7)
        angles.append(_angle_entry(
            "LL", "Lumbar Lordosis", LL, "#f472b6",
            [_seg(P1 - HW * e1, P1 + HW * e1), _seg(P7 - HW * e7, P7 + HW * e7),
             _seg(mid, mid + 0.5 * RAY * n1s), _seg(mid, mid + 0.5 * RAY * n7s)],
            (mid, mid + n1s, mid + n7s), mid + 0.5 * (n1s + n7s) * 46))

    return {"sagittal_normal": [round(float(x), 4) for x in lr],
            "plane_origin": _p(origin), "angles": angles, "points": points}


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
    ct = caff = None
    if args.ct:
        ct, caff = load_ct(args.ct)
        if ct.shape != seg.shape:
            raise SystemExit("CT and label grids differ; resample first")

    geom = build_geometry(seg, laff)
    summary = metrics.spinopelvic_summary_from_label(seg, laff, case_id=args.case_id)

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
    process(p.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
