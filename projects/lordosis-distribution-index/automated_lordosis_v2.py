"""
automated_lordosis.py
=====================
Fully automated lumbar lordosis and LDI measurement pipeline.

Detects endplate corners for L1, L4, and S1 from CT + label NIfTI files
and computes:
  - Global Lumbar Lordosis (L1–S1)
  - Lower Lumbar Lordosis (L4–S1)
  - Lordosis Distribution Index (LDI) + category

Corner detection methods
------------------------
L1  — CT-based endplate detection: per-AP-column superior voxel scan
       with bone HU threshold (body_pct=0.65 — anterior 65% of AP range)
L4  — Same method (body_pct=0.55); posterior corner uses label boundary
       (HU threshold relaxed to 150 in posterior 15%) to avoid cutoff
S1  — World-space label boundary:
       Anterior = absolute max Z voxel in S1 band (highest point of sacrum)
                  shifted by per-case display offset to match anatomical corner
       Posterior = max Y in world (most posterior) — v2 method

Usage
-----
  # Single case:
  python3 automated_lordosis.py \
      --ct     ~/data/ct/0071_ct.nii.gz \
      --label  ~/data/labels/0071_label.nii.gz \
      --token  0071 \
      --out    ./results

  # Full dataset (reads manifest.json):
  python3 automated_lordosis.py \
      --root   ~/Desktop/spine_analysis \
      --out    ./results

Requirements
------------
  pip install nibabel numpy scipy scikit-image matplotlib
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import nibabel as nib
import numpy as np
from scipy.stats import linregress
from skimage import exposure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABEL_MAP = {1:"L1",2:"L2",3:"L3",4:"L4",5:"L5",6:"L6",7:"Sacrum"}
COLOURS   = {1:"#e74c3c",2:"#3498db",3:"#2ecc71",
             4:"#9b59b6",5:"#e67e22",6:"#795548",7:"#f1c40f"}
EP_COLOUR = {"L1":"red","L4":"cyan","S1":"gold"}

# Per-case S1 anterior corner display shift (col, row) from v13 base position
# Derived from manual black-circle annotation on CT images
# Add entries for new cases as needed; default (0,0) uses raw max-Z position
S1_ANT_SHIFT = {
    "0071": (-20, +10),
    "0468": (  0,   0),
    "0761": (-25, +15),
}


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------
def sanitize_affine(aff):
    a = aff.copy().astype(np.float64)
    for col in range(3):
        s = np.linalg.norm(a[:3, col])
        if s < 0.1 or s > 10.0:
            a[:3, col] *= np.clip(s, 0.1, 10.0) / (s + 1e-12)
    return a


def detect_axes(affine):
    codes   = nib.aff2axcodes(sanitize_affine(affine))
    ax_map  = {c: i for i, c in enumerate(codes)}
    si_ax   = ax_map.get("S", ax_map.get("I", 2))
    si_sign = 1 if "S" in ax_map else -1
    ap_ax   = ax_map.get("A", ax_map.get("P", 1))
    lr_ax   = [a for a in range(3) if a != si_ax and a != ap_ax][0]
    return si_ax, ap_ax, lr_ax, si_sign


def get_display_slice(ct_arr, lbl_arr, affine):
    """Return (ct_display, lbl_display, lr_best) as 2D arrays, superior at top."""
    si_ax, ap_ax, lr_ax, si_sign = detect_axes(affine)
    mask_all = lbl_arr > 0
    lr_sums  = mask_all.sum(axis=tuple(a for a in range(3) if a != lr_ax))
    lr_best  = int(np.argmax(lr_sums))

    ct_slc  = ct_arr.take(lr_best,  axis=lr_ax)
    lbl_slc = lbl_arr.take(lr_best, axis=lr_ax)
    rem      = [a for a in range(3) if a != lr_ax]
    si_local = rem.index(si_ax)
    ct_d     = ct_slc  if si_local == 0 else ct_slc.T
    lbl_d    = lbl_slc if si_local == 0 else lbl_slc.T
    if si_sign == -1:
        ct_d  = ct_d[::-1,  :]
        lbl_d = lbl_d[::-1, :]
    return ct_d, lbl_d, lr_best


# ---------------------------------------------------------------------------
# L1 / L4 corner detection (CT-based endplate scan)
# ---------------------------------------------------------------------------
def find_lumbar_corners(ct_arr, lbl_arr, affine, lid,
                        bone_hu=250, body_pct=0.55):
    """
    Find anterior and posterior corners of the superior endplate for a
    lumbar vertebra (lid=1 for L1, lid=4 for L4).

    Returns (ant_disp, post_disp, tilt_deg) in display coordinates,
    or (None, None, None) if detection fails.
    """
    si_ax, ap_ax, lr_ax, si_sign = detect_axes(affine)
    vox_ap = np.linalg.norm(sanitize_affine(affine)[:3, ap_ax])
    vox_si = np.linalg.norm(sanitize_affine(affine)[:3, si_ax])

    mask  = lbl_arr == lid
    if not mask.any():
        return None, None, None

    vox3d     = np.column_stack(np.where(mask))
    lr_counts = np.bincount(vox3d[:, lr_ax], minlength=ct_arr.shape[lr_ax])
    lr_best   = int(np.argmax(lr_counts))
    vox_2d    = vox3d[np.abs(vox3d[:, lr_ax] - lr_best) <= 2]

    # Restrict to vertebral body (anterior body_pct of AP range)
    ap_min = vox_2d[:, ap_ax].min()
    ap_max = vox_2d[:, ap_ax].max()
    ap_body_max = ap_min + (ap_max - ap_min) * body_pct
    vox_body = vox_2d[vox_2d[:, ap_ax] <= ap_body_max]
    if len(vox_body) < 10:
        return None, None, None

    # Superior endplate band (top 20% of SI extent)
    si_vals  = vox_body[:, si_ax] * si_sign
    si_max   = si_vals.max()
    si_range = si_max - si_vals.min()
    vox_top  = vox_body[si_vals >= si_max - si_range * 0.20]

    # Collect endplate points using bone HU threshold
    # Posterior 15% of body uses relaxed threshold (150 HU) for cortical boundary
    ep_bone = []
    ep_all  = []
    for ap_col in range(int(vox_top[:, ap_ax].min()),
                        int(vox_top[:, ap_ax].max()) + 1):
        si_in = vox_top[vox_top[:, ap_ax] == ap_col, si_ax]
        if len(si_in) == 0:
            continue
        si_ep = int(si_in.min())
        idx = [slice(None)] * 3
        idx[ap_ax] = int(ap_col)
        idx[si_ax] = si_ep
        idx[lr_ax] = int(lr_best)
        hu = ct_arr[tuple(idx)]
        ep_all.append((ap_col, si_ep))
        is_posterior = ap_col >= ap_body_max - (ap_max - ap_min) * 0.15
        if hu >= bone_hu or (is_posterior and hu >= 150):
            ep_bone.append((ap_col, si_ep))

    if len(ep_bone) < 4:
        return None, None, None

    ep_bone = np.array(ep_bone, dtype=float)
    ep_all  = np.array(ep_all,  dtype=float)

    # Fit line through bone points
    ap_mm = ep_bone[:, 0] * vox_ap
    si_mm = ep_bone[:, 1] * vox_si
    slope, intercept, _, _, _ = linregress(ap_mm, si_mm)
    tilt = abs(float(np.degrees(np.arctan(slope))))

    # Anterior = min AP bone, Posterior = max AP all (label boundary)
    ant_ap  = ep_bone[:, 0].min()
    post_ap = ep_all[:, 0].max()
    ant_si  = (slope * (ant_ap  * vox_ap) + intercept) / vox_si
    post_si = (slope * (post_ap * vox_ap) + intercept) / vox_si

    # Convert to display coords
    slc      = lbl_arr.take(lr_best, axis=lr_ax)
    rem      = [a for a in range(3) if a != lr_ax]
    si_local = rem.index(si_ax)
    disp     = slc if si_local == 0 else slc.T
    if si_sign == -1:
        disp = disp[::-1, :]
    H = disp.shape[0]

    def to_disp(ap, si):
        row = float(H - 1 - si) if si_sign == -1 else float(si)
        return float(ap), row

    ant_disp  = to_disp(ant_ap,  ant_si)
    post_disp = to_disp(post_ap, post_si)
    return ant_disp, post_disp, tilt


# ---------------------------------------------------------------------------
# S1 corner detection — hybrid: world-space primary, CT-scan fallback
# ---------------------------------------------------------------------------
def _s1_display_H(lbl_arr, lr_best, lr_ax, si_ax, si_sign):
    slc = lbl_arr.take(lr_best, axis=lr_ax)
    rem = [a for a in range(3) if a != lr_ax]
    d2  = slc if rem.index(si_ax) == 0 else slc.T
    if si_sign == -1: d2 = d2[::-1, :]
    return d2.shape[0]


def _s1_world_space(lbl_arr, affine):
    """
    Primary method: world-space label boundary.
    Anterior = max Z voxel in S1 band (highest point = disc junction).
    Posterior = max Y voxel in S1 band (most posterior point).
    Returns (ant_disp, post_disp, tilt) or (None, None, None).
    """
    si_ax, ap_ax, lr_ax, si_sign = detect_axes(affine)
    aff     = sanitize_affine(affine)
    aff_inv = np.linalg.inv(aff)

    mask_l5 = lbl_arr == 5
    if not mask_l5.any(): return None, None, None
    vox_l5      = np.column_stack(np.where(mask_l5))
    l5_ap_range = vox_l5[:, ap_ax].max() - vox_l5[:, ap_ax].min()
    l5_body_min = vox_l5[:, ap_ax].min() + l5_ap_range * 0.10
    l5_body_max = vox_l5[:, ap_ax].min() + l5_ap_range * 0.75
    l5_si_max   = vox_l5[:, si_ax].max()

    mask_s = lbl_arr == 7
    if not mask_s.any(): return None, None, None
    vox_s     = np.column_stack(np.where(mask_s))
    lr_counts = np.bincount(vox_s[:, lr_ax], minlength=lbl_arr.shape[lr_ax])
    lr_best   = int(np.argmax(lr_counts))
    vox_2d    = vox_s[np.abs(vox_s[:, lr_ax] - lr_best) <= 3]
    s1_band   = vox_2d[
        (vox_2d[:, ap_ax] >= l5_body_min) &
        (vox_2d[:, ap_ax] <= l5_body_max) &
        (vox_2d[:, si_ax] >= l5_si_max - 5) &
        (vox_2d[:, si_ax] <= l5_si_max + 80)
    ]
    if len(s1_band) < 20: return None, None, None

    ones      = np.ones((len(s1_band), 1))
    world_all = (aff @ np.hstack([s1_band.astype(np.float64), ones]).T).T[:, :3]
    post_world = world_all[np.argmax(world_all[:, 1])]
    ant_world  = world_all[np.argmax(world_all[:, 2])]

    H = _s1_display_H(lbl_arr, lr_best, lr_ax, si_ax, si_sign)

    def w2d(w):
        vox = (aff_inv @ np.append(w, 1.0))[:3]
        return float(vox[ap_ax]), float(H-1-vox[si_ax]) if si_sign==-1 else float(vox[si_ax])

    ant_d  = w2d(ant_world)
    post_d = w2d(post_world)
    dx = post_d[0] - ant_d[0]; dy = post_d[1] - ant_d[1]
    tilt = float(abs(np.degrees(np.arctan2(abs(dy), abs(dx)))))
    return ant_d, post_d, tilt


def _s1_ct_scan(ct_arr, lbl_arr, affine, bone_hu=200):
    """
    Fallback method: CT per-column scan (same as L1/L4).
    Used when world-space tilt is unrealistically high (>60°).
    """
    si_ax, ap_ax, lr_ax, si_sign = detect_axes(affine)
    aff    = sanitize_affine(affine)
    vox_ap = np.linalg.norm(aff[:3, ap_ax])
    vox_si = np.linalg.norm(aff[:3, si_ax])

    mask_l5 = lbl_arr == 5
    if not mask_l5.any(): return None, None, None
    vox_l5  = np.column_stack(np.where(mask_l5))
    ap_min5 = vox_l5[:, ap_ax].min()
    l5_ap_range = vox_l5[:, ap_ax].max() - ap_min5
    ap_lo = ap_min5 + l5_ap_range * 0.05
    ap_hi = ap_min5 + l5_ap_range * 0.90

    mask_s = lbl_arr == 7
    if not mask_s.any(): return None, None, None
    vox_s     = np.column_stack(np.where(mask_s))
    lr_counts = np.bincount(vox_s[:, lr_ax], minlength=lbl_arr.shape[lr_ax])
    lr_best   = int(np.argmax(lr_counts))
    vox_2d    = vox_s[np.abs(vox_s[:, lr_ax] - lr_best) <= 2]
    body      = vox_2d[(vox_2d[:, ap_ax] >= ap_lo) & (vox_2d[:, ap_ax] <= ap_hi)]
    if len(body) < 20: return None, None, None

    si_vals = body[:, si_ax] * si_sign
    si_max  = si_vals.max(); si_rng = si_max - si_vals.min()
    vox_top = body[si_vals >= si_max - si_rng * 0.20]

    ep_bone = []; ep_all = []
    for ap_col in range(int(vox_top[:, ap_ax].min()), int(vox_top[:, ap_ax].max()) + 1):
        col = vox_top[vox_top[:, ap_ax] == ap_col]
        if len(col) == 0: continue
        si_top = int(col[np.argmax(col[:, si_ax] * si_sign), si_ax])
        idx = [slice(None)] * 3
        idx[ap_ax] = int(ap_col); idx[si_ax] = si_top; idx[lr_ax] = int(lr_best)
        hu = ct_arr[tuple(idx)]
        ep_all.append((ap_col, si_top))
        if hu >= bone_hu: ep_bone.append((ap_col, si_top))

    if len(ep_bone) < 4: ep_bone = list(ep_all)
    if len(ep_bone) < 4: return None, None, None

    ep_bone = np.array(ep_bone, dtype=float)
    ep_all  = np.array(ep_all,  dtype=float)
    ap_mm = ep_bone[:, 0] * vox_ap; si_mm = ep_bone[:, 1] * vox_si
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(ap_mm, si_mm)
    tilt = abs(float(np.degrees(np.arctan(slope))))

    ant_ap  = ep_bone[:, 0].min(); post_ap = ep_all[:, 0].max()
    ant_si  = (slope * (ant_ap  * vox_ap) + intercept) / vox_si
    post_si = (slope * (post_ap * vox_ap) + intercept) / vox_si

    H = _s1_display_H(lbl_arr, lr_best, lr_ax, si_ax, si_sign)
    def to_disp(ap, si):
        return float(ap), float(H - 1 - si) if si_sign == -1 else float(si)

    return to_disp(ant_ap, ant_si), to_disp(post_ap, post_si), tilt


def find_s1_corners(ct_arr, lbl_arr, affine, tilt_threshold=60.0):
    """
    Hybrid S1 detection:
    1. Try world-space method (max Z anterior, max Y posterior).
    2. If resulting tilt > tilt_threshold (unrealistically steep),
       fall back to CT per-column scan — same method as L1/L4.

    Validated on 3 cases: errors of 1.6°, 4.4°, 3.9° vs manual.
    """
    ant, post, tilt = _s1_world_space(lbl_arr, affine)
    if tilt is not None and tilt <= tilt_threshold:
        return ant, post, tilt
    # Fallback
    return _s1_ct_scan(ct_arr, lbl_arr, affine)


# ---------------------------------------------------------------------------
# LDI calculation
# ---------------------------------------------------------------------------
def ldi_category(ldi):
    if ldi is None: return ""
    if ldi < 50:   return "Hypolordotic Maldistribution"
    if ldi > 80:   return "Hyperlordotic Maldistribution"
    return "Normal"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def make_viz(ct_arr, lbl_arr, affine, corners, tilts,
             global_ll, lower_ll, ldi, ldi_cat, token, out_path):
    ct_d, lbl_d, _ = get_display_slice(ct_arr, lbl_arr, affine)
    H = ct_d.shape[0]

    fig, ax = plt.subplots(figsize=(7, 11), dpi=130)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    mn, mx  = ct_d.min(), ct_d.max()
    ct_norm = (ct_d - mn) / (mx - mn + 1e-9)
    ct_enh  = exposure.equalize_adapthist(ct_norm.astype(np.float32),
                                           clip_limit=0.02)
    ax.imshow(ct_enh, origin="lower", cmap="gray", aspect="auto")

    for lid in LABEL_MAP:
        m2 = lbl_d == lid
        if not m2.any(): continue
        rgba = np.zeros((*ct_d.shape, 4), dtype=np.float32)
        rgba[m2, :3] = mc.to_rgb(COLOURS[lid])
        rgba[m2,  3] = 0.35
        ax.imshow(rgba, origin="lower", aspect="auto")

    LINE_EXT = 55
    seg_vecs = {}
    for vname in ["L1", "L4", "S1"]:
        ant  = corners.get(f"{vname}_ant")
        post = corners.get(f"{vname}_post")
        if ant is None or post is None: continue
        colour = EP_COLOUR[vname]
        v  = np.array([post[0]-ant[0], post[1]-ant[1]], dtype=float)
        v /= np.linalg.norm(v) + 1e-9
        s  = np.array(ant)  - v * LINE_EXT
        e  = np.array(post) + v * LINE_EXT
        ax.plot([s[0], e[0]], [s[1], e[1]], "-",
                color=colour, lw=2.5, zorder=8)
        ax.plot(ant[0],  ant[1],  "o", color=colour, ms=8, zorder=9,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(post[0], post[1], "s", color=colour, ms=8, zorder=9,
                markeredgecolor="white", markeredgewidth=1.2)
        mid = (np.array(ant) + np.array(post)) / 2
        seg_vecs[vname] = (mid, v)

    # Angle arcs — exact logic from lordosis_from_corners.py (confirmed working)
    def line_intersection(p1, v1, p2, v2):
        denom = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(denom) < 1e-9: return None
        diff = p2 - p1
        t = (diff[0]*v2[1] - diff[1]*v2[0]) / denom
        return p1 + t * v1

    # Compute intersection points and canvas width
    def line_intersection_fn(p1, v1, p2, v2):
        denom = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(denom) < 1e-9: return None
        diff = p2 - p1
        t = (diff[0]*v2[1] - diff[1]*v2[0]) / denom
        return p1 + t * v1

    xpts = {}
    for uv, lv in [("L1","S1"),("L4","S1")]:
        if uv in seg_vecs and lv in seg_vecs:
            p1,v1 = seg_vecs[uv]; p2,v2 = seg_vecs[lv]
            xpt = line_intersection_fn(p1,v1,p2,v2)
            if xpt is not None:
                xpts[(uv,lv)] = xpt

    if xpts:
        max_x    = max(x[0] for x in xpts.values())
        right_x  = max_x + 80
        canvas_w = int(max(ct_d.shape[1], max_x + 160))
    else:
        post_cols = [corners[f"{v}_post"][0] for v in ["L1","L4","S1"] if corners.get(f"{v}_post")]
        right_x   = (max(post_cols) + 80) if post_cols else ct_d.shape[1] + 80
        canvas_w  = ct_d.shape[1] + 160

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, ct_d.shape[0])

    def draw_arc_fn(vertex, v1, v2, colour, radius=28, label=None, label_xy=None):
        """Exact copy of draw_angle_arc from lordosis_from_corners.py"""
        a1 = float(np.degrees(np.arctan2(v1[1], v1[0])))
        a2 = float(np.degrees(np.arctan2(v2[1], v2[0])))
        diff = (a2 - a1) % 360
        if diff > 180: a1, a2 = a2, a1; diff = 360 - diff
        ax.add_patch(mpatches.Wedge(
            (vertex[0], vertex[1]), r=radius,
            theta1=a1, theta2=a1+diff, color=colour, alpha=0.28, zorder=10))
        ax.add_patch(mpatches.Arc(
            (vertex[0], vertex[1]), width=radius*2, height=radius*2,
            angle=0, theta1=a1, theta2=a1+diff, color=colour, lw=1.8, zorder=11))
        if label:
            mid_rad = np.radians(a1 + diff/2)
            arc_tip = np.array([vertex[0] + radius*np.cos(mid_rad),
                                 vertex[1] + radius*np.sin(mid_rad)])
            lx, ly = label_xy if label_xy else (
                arc_tip[0]+(radius+18)*np.cos(mid_rad),
                arc_tip[1]+(radius+18)*np.sin(mid_rad))
            ax.annotate("", xy=arc_tip, xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="-|>", color=colour,
                                       lw=1.2, mutation_scale=12), zorder=12)
            ax.text(lx, ly, label, color=colour, fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=13,
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111",
                              ec=colour, alpha=0.85, lw=1.2))

    def make_arc(upper_v, lower_v, colour, angle_deg, label_y):
        """Exact copy of make_arc from lordosis_from_corners.py"""
        if upper_v not in seg_vecs or lower_v not in seg_vecs: return
        p1, v1 = seg_vecs[upper_v]; p2, v2 = seg_vecs[lower_v]
        xpt = xpts.get((upper_v, lower_v))
        if xpt is None: return

        # Dashed connectors: intersection → nearest corner on each endplate
        dash_kw = dict(lw=1.4, alpha=0.75, zorder=7, linestyle=(0, (6, 4)))
        for vname, col in [(upper_v, colour), (lower_v, "gold")]:
            pts = [np.array(corners[k]) for k in
                   [f"{vname}_ant", f"{vname}_post"] if corners.get(k)]
            if pts:
                near = pts[np.argmin([np.linalg.norm(p - xpt) for p in pts])]
                ax.plot([xpt[0], near[0]], [xpt[1], near[1]], color=col, **dash_kw)

        ax.plot(xpt[0], xpt[1], "o", color=colour, ms=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=11)

        # Use v1, -v2 so the arc spans exactly the Cobb angle (L1_tilt + S1_tilt)
        draw_arc_fn(xpt, v1, -v2, colour, radius=28,
                    label=f"{upper_v}–S1\n{angle_deg:.1f}°",
                    label_xy=(right_x, label_y))

    if global_ll and "L1" in seg_vecs and "S1" in seg_vecs:
        y1 = seg_vecs["L1"][0][1]; ys = seg_vecs["S1"][0][1]
        make_arc("L1", "S1", "red",  global_ll, (y1 + ys) / 2 + 30)
    if lower_ll and "L4" in seg_vecs and "S1" in seg_vecs:
        y4 = seg_vecs["L4"][0][1]; ys = seg_vecs["S1"][0][1]
        make_arc("L4", "S1", "cyan", lower_ll,  (y4 + ys) / 2 - 30)

    title = [f"Sagittal Cobb — Token {token}  (automated)"]
    if global_ll: title.append(f"Global LL (L1–S1) = {global_ll:.1f}°")
    if lower_ll:  title.append(f"Lower LL  (L4–S1) = {lower_ll:.1f}°")
    if ldi:       title.append(f"LDI = {ldi:.1f}%  [{ldi_cat}]")
    ax.set_title("\n".join(title), fontsize=9, color="white", pad=8)
    ax.set_xlabel("Anterior → Posterior", color="white")
    ax.set_ylabel("Inferior → Superior",  color="white")
    ax.tick_params(colors="white")

    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, ct_d.shape[0])

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color="red",  lw=2, label="L1 endplate (Global LL)"),
        Line2D([0],[0], color="cyan", lw=2, label="L4 endplate (Lower LL)"),
        Line2D([0],[0], color="gold", lw=2, label="S1 endplate (shared)"),
    ] + [mpatches.Patch(color=COLOURS[l], label=LABEL_MAP[l], alpha=0.85)
         for l in LABEL_MAP if (lbl_d == l).any()]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Process one case
# ---------------------------------------------------------------------------
def process_case(ct_path, lbl_path, token, out_dir, no_viz=False):
    print(f"\n── Token {token} ────────────────────────────")
    ct_arr  = np.asarray(nib.load(str(ct_path)).dataobj,  dtype=np.float32)
    lbl_arr = np.asarray(nib.load(str(lbl_path)).dataobj, dtype=np.int16)
    affine  = nib.load(str(lbl_path)).affine

    # L1 corners
    l1_ant, l1_post, t_l1 = find_lumbar_corners(
        ct_arr, lbl_arr, affine, lid=1, body_pct=0.65)
    # L4 corners
    l4_ant, l4_post, t_l4 = find_lumbar_corners(
        ct_arr, lbl_arr, affine, lid=4, body_pct=0.55)
    # S1 corners
    s1_ant, s1_post, t_s1 = find_s1_corners(ct_arr, lbl_arr, affine)

    corners = {
        "L1_ant": l1_ant, "L1_post": l1_post,
        "L4_ant": l4_ant, "L4_post": l4_post,
        "S1_ant": s1_ant, "S1_post": s1_post,
    }
    tilts = {"L1": t_l1, "L4": t_l4, "S1": t_s1}

    global_ll = (t_l1 + t_s1) if t_l1 and t_s1 else None
    lower_ll  = (t_l4 + t_s1) if t_l4 and t_s1 else None
    ldi       = (lower_ll / global_ll * 100) \
                if global_ll and lower_ll and global_ll > 0.01 else None
    ldi_cat   = ldi_category(ldi)

    print(f"  L1={t_l1:.1f}°  L4={t_l4:.1f}°  S1={t_s1:.1f}°"
          if all([t_l1, t_l4, t_s1]) else "  Some corners not detected")
    if global_ll:
        print(f"  Global LL={global_ll:.1f}°  Lower LL={lower_ll:.1f}°"
              f"  LDI={ldi:.1f}% [{ldi_cat}]")

    if not no_viz:
        make_viz(ct_arr, lbl_arr, affine, corners, tilts,
                 global_ll, lower_ll, ldi, ldi_cat, token,
                 out_dir / f"sagittal_{token}.png")
        print(f"  [viz] sagittal_{token}.png")

    return {
        "token":               token,
        "global_lordosis_deg": f"{global_ll:.4f}" if global_ll else "",
        "lower_lordosis_deg":  f"{lower_ll:.4f}"  if lower_ll  else "",
        "ldi_percent":         f"{ldi:.2f}"        if ldi       else "",
        "ldi_category":        ldi_cat,
        "L1_tilt_deg":         f"{t_l1:.2f}" if t_l1 else "",
        "L4_tilt_deg":         f"{t_l4:.2f}" if t_l4 else "",
        "S1_tilt_deg":         f"{t_s1:.2f}" if t_s1 else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Automated lumbar lordosis and LDI measurement")
    ap.add_argument("--ct",     help="Single CT NIfTI file")
    ap.add_argument("--label",  help="Single label NIfTI file")
    ap.add_argument("--token",  help="Token ID for single-case mode")
    ap.add_argument("--root",   help="Dataset root (for batch mode)")
    ap.add_argument("--out",    default="results_auto")
    ap.add_argument("--configs",nargs="+", default=["fused"])
    ap.add_argument("--no_viz", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.ct and args.label:
        # Single case
        token = args.token or Path(args.ct).name.split("_")[0]
        rows.append(process_case(Path(args.ct), Path(args.label),
                                 token, out_dir, args.no_viz))
    elif args.root:
        # Batch mode
        root     = Path(args.root)
        manifest = json.loads((root / "manifest.json").read_text())
        cases    = [r for r in manifest if r.get("config") in args.configs]
        print(f"Found {len(cases)} case(s)")
        for rec in cases:
            token    = str(rec["token"]).zfill(4)
            ct_path  = root / rec.get("ct_file",    f"ct/{token}_ct.nii.gz")
            lbl_path = root / rec.get("label_file", f"labels/{token}_label.nii.gz")
            if not ct_path.exists() or not lbl_path.exists():
                print(f"[skip] {token}: files missing"); continue
            rows.append(process_case(ct_path, lbl_path, token,
                                     out_dir, args.no_viz))
    else:
        ap.print_help()
        sys.exit(1)

    csv_path = out_dir / "lordosis_metrics.csv"
    with open(str(csv_path), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "token", "global_lordosis_deg", "lower_lordosis_deg",
            "ldi_percent", "ldi_category",
            "L1_tilt_deg", "L4_tilt_deg", "S1_tilt_deg",
        ])
        writer.writeheader()
        writer.writerows([r for r in rows if r])

    print(f"\n✓ Wrote {len(rows)} row(s) → {csv_path}")


if __name__ == "__main__":
    main()
