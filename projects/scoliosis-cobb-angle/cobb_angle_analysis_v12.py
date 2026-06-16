"""
cobb_angle_analysis_v3.py
=========================
Coronal Cobb angle — faithful implementation of the clinical gold standard.

CLINICAL METHOD (PA X-ray, Scoliosis Research Society standard):
  1. Identify ALL vertebral endplates in the lumbar spine.
  2. For each vertebra, fit a plane to the ACTUAL endplate surface voxels
     (top 15% of voxels by SI height = superior endplate;
      bottom 15% = inferior endplate).
  3. Use minimum-variance PCA on the surface points — NOT whole-body PCA —
     to get the true endplate normal.
  4. AUTO-DETECT the most-tilted pair: try every adjacent vertebra combination
     and find the pair that produces the largest coronal angle. This matches
     what a radiologist does — they do NOT always use L1 and L5.
  5. Report the Cobb angle between those two endplate lines, plus L1-L5 as a
     fixed reference and L1-Sacrum as a secondary metric.

  The angle is between the two endplate LINES (perpendicular to the normals),
  mathematically equivalent to the intersection-of-perpendiculars method used
  on PA X-rays.

Labels: 1=L1  2=L2  3=L3  4=L4  5=L5  6=L6/LSTV  7=Sacrum

Outputs:
  cobb_angles.csv              One row per case
  cobb_viz_<token>.png         Coronal view with the auto-detected Cobb pair highlighted

Usage:
  python cobb_angle_analysis_v3.py --hub anonymous-neurips-ED/CTSpinoPelvic1K-Sample
  python cobb_angle_analysis_v3.py --root /path/to/local/data
"""

import argparse
import csv
import logging
import math
import sys
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL_MAP   = {1:"L1",2:"L2",3:"L3",4:"L4",5:"L5",6:"L6",7:"Sacrum"}
LUMBAR_LBLS = list(range(1, 8))

# ── Colours for visualisation ─────────────────────────────────────────────────
COLORS = {
    1: "#e63946",   # L1  red
    2: "#f4a261",   # L2  orange
    3: "#2a9d8f",   # L3  teal
    4: "#457b9d",   # L4  steel blue
    5: "#a8dadc",   # L5  light blue
    6: "#6d6875",   # L6  purple
    7: "#e9c46a",   # Sacrum  gold
}

# ── Scoliosis grading (Scoliosis Research Society standard) ───────────────────
SCOLIOSIS_GRADES = [
    ( 0,  10, "Normal",      "< 10 deg — no scoliosis"),
    (10,  25, "Mild",        "10–25 deg — observe, serial imaging"),
    (25,  40, "Moderate",    "25–40 deg — bracing indicated"),
    (40,  60, "Severe",      "40–60 deg — surgical candidate"),
    (60, 999, "Very Severe", "> 60 deg — surgery strongly indicated"),
]

def grade_scoliosis(angle_deg_val):
    try:
        a = float(angle_deg_val)
    except (TypeError, ValueError):
        return "N/A", "N/A"
    for lo, hi, grade, desc in SCOLIOSIS_GRADES:
        if lo <= a < hi:
            return grade, desc
    return "N/A", "N/A"


# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════

def voxels_to_world(vox, affine):
    """(N,3) integer voxel indices → (N,3) world-mm coordinates."""
    n = vox.shape[0]
    h = np.ones((n, 4), dtype=np.float64)
    h[:, :3] = vox
    return (affine @ h.T).T[:, :3]


def get_si_axis(affine):
    """
    Return the index (0,1,2) of the voxel axis most aligned with the
    world Superior–Inferior direction, and its sign (+1 = voxel increases
    toward superior, -1 = voxel increases toward inferior).
    World S-I = world axis 2 (z) in RAS convention.
    """
    R = affine[:3, :3]
    # Which voxel axis is most aligned with world-z (Superior)?
    col = np.argmax(np.abs(R[2, :]))
    sign = float(np.sign(R[2, col]))
    return int(col), sign


def largest_connected_cluster(world_pts, gap_mm=8.0):
    """
    Find the largest spatially connected cluster of points in world-mm space.

    This solves the fragmented segmentation problem: when a vertebra's
    label mask is split into multiple disconnected blobs (e.g. L4 and L5
    in spine_only configs), taking the top/bottom fraction of ALL points
    lands on a stray fragment rather than the actual vertebral body.

    Algorithm: single-linkage clustering on the SI axis only.
      - Sort points by SI coordinate (world-z).
      - Any gap > gap_mm between consecutive SI values marks a cluster break.
      - Return the points belonging to the largest cluster.

    Using SI-only gaps (not 3D distance) is intentional: fragments from
    the same vertebra label tend to be separated vertically, not laterally.
    A 3D distance threshold would be too slow on large point clouds and
    would incorrectly split wide vertebral bodies.

    Args:
      world_pts : (N,3) world-mm coordinates
      gap_mm    : SI gap threshold to separate clusters (default 8mm — roughly
                  half a lumbar disc height, so adjacent vertebrae are never
                  merged, but fragments of the same vertebra are)
    """
    if len(world_pts) == 0:
        return world_pts

    # Sort by SI (world-z)
    order    = np.argsort(world_pts[:, 2])
    sorted_z = world_pts[order, 2]

    # Find cluster boundaries: positions where gap > gap_mm
    gaps      = np.diff(sorted_z)
    breaks    = np.where(gaps > gap_mm)[0] + 1   # indices where new cluster starts
    starts    = np.concatenate([[0], breaks])
    ends      = np.concatenate([breaks, [len(world_pts)]])

    # Pick the largest cluster by point count
    sizes     = ends - starts
    best      = int(np.argmax(sizes))
    cluster_idx = order[starts[best]:ends[best]]

    return world_pts[cluster_idx]


def isolate_endplate_surface(world_pts, which="superior", frac=0.15, min_pts=40):
    """
    Extract only the endplate surface voxels from a vertebral point cloud.

    Three-pass approach — handles badly fragmented segmentations robustly:

      Pass 1 — Largest cluster:
        Find the biggest spatially-connected blob in the SI axis. This is
        the actual vertebral body. Detached fragments (e.g. split L4/L5
        labels in spine_only scans) are discarded entirely before any
        surface selection happens. This is the key fix for floating lines.

      Pass 2 — IQR trim:
        Within the main cluster, apply a Tukey fence to remove any residual
        stray voxels at the top/bottom edges of the cluster.

      Pass 3 — Surface fraction:
        Take the top/bottom `frac` fraction of the cleaned cluster as the
        endplate surface layer for plane fitting.

    Args:
      world_pts : (N,3) world-mm coordinates
      which     : "superior" or "inferior"
      frac      : fraction of cleaned points to use as surface (default 15%)
      min_pts   : minimum surface points (fallback if any pass over-clips)
    """
    if len(world_pts) < min_pts:
        return world_pts

    # ── Pass 1: largest connected cluster in SI axis ──────────────────────────
    main_cluster = largest_connected_cluster(world_pts, gap_mm=8.0)
    if len(main_cluster) < min_pts:
        main_cluster = world_pts   # fallback

    # ── Pass 2: IQR trim within the main cluster ──────────────────────────────
    si_vals = main_cluster[:, 2]
    q1  = float(np.percentile(si_vals, 25))
    q3  = float(np.percentile(si_vals, 75))
    iqr = q3 - q1

    if iqr > 0:
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        keep    = (si_vals >= lo) & (si_vals <= hi)
        cleaned = main_cluster[keep]
    else:
        cleaned = main_cluster

    if len(cleaned) < min_pts:
        cleaned = main_cluster   # fallback

    # ── Pass 3: surface fraction from cleaned cluster ─────────────────────────
    si_clean = cleaned[:, 2]
    n_take   = max(min_pts, int(len(cleaned) * frac))
    n_take   = min(n_take, len(cleaned))

    if which == "superior":
        idx = np.argpartition(si_clean, -n_take)[-n_take:]
    else:
        idx = np.argpartition(si_clean,  n_take)[: n_take]

    return cleaned[idx]


def fit_plane_to_surface(surface_pts):
    """
    Fit a plane to a set of (near-flat) surface points using PCA.

    Key insight: for a flat surface the normal is the direction of
    MINIMUM variance (the thin axis of the point cloud), not maximum.
    Whole-body PCA picks maximum variance which runs along the SI axis
    of the vertebra — that is NOT the endplate normal. This function
    correctly picks the minimum-variance axis.

    Returns:
      centroid : (3,) mean of the surface points
      normal   : (3,) unit normal pointing cranially (SI+)
    """
    centroid = surface_pts.mean(axis=0)
    _, s, Vt = np.linalg.svd(surface_pts - centroid, full_matrices=False)
    # Minimum singular value = flattest direction = endplate normal
    normal = Vt[np.argmin(s)].copy()
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def endplate_normal_coronal(world_pts, affine, which="superior"):
    """
    Compute the endplate normal projected onto the coronal plane,
    using only the actual endplate surface voxels.

    Clinical Cobb method:
      L1 -> superior endplate (top surface, faces cranially)
      L5 -> inferior endplate (bottom surface, faces caudally)

    Returns:
      proj          : (2,) [RL, SI] unit normal in the coronal plane
      body_centroid : (3,) centroid of the full vertebral body (for labels)
      ep_centroid   : (3,) centroid of the endplate surface (for line anchor)
    """
    body_centroid = world_pts.mean(axis=0)
    surface_pts = isolate_endplate_surface(world_pts, which=which)
    ep_centroid, normal = fit_plane_to_surface(surface_pts)
    proj = np.array([normal[0], normal[2]])
    n_len = np.linalg.norm(proj)
    if n_len > 1e-9:
        proj /= n_len
    else:
        proj = np.array([0.0, 1.0])
    return proj, body_centroid, ep_centroid



def cobb_angle(proj1, proj2):
    """
    Angle in degrees between two 2D coronal-plane endplate normal vectors.
    Always returns the smaller of the two supplementary angles (0–90°).
    """
    n1 = np.linalg.norm(proj1)
    n2 = np.linalg.norm(proj2)
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    ct = np.clip(np.dot(proj1, proj2) / (n1 * n2), -1.0, 1.0)
    a = math.degrees(math.acos(ct))
    return min(a, 180.0 - a)


# ═════════════════════════════════════════════════════════════════════════════
# PER-CASE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def analyse_case(label_vol, affine, token):
    """
    Compute all Cobb metrics for one case, using the true clinical method:
      - Fit endplate normals to actual surface voxels (not whole-body PCA)
      - Auto-detect the most-tilted vertebra pair (true Cobb definition)
      - Also report fixed L1-L5 and L1-Sacrum for reference

    pca_data stores BOTH endplates per vertebra:
      pca_data[lid] = {
        "body_centroid": (3,),
        "sup": (proj_2d, ep_centroid),   # superior endplate
        "inf": (proj_2d, ep_centroid),   # inferior endplate
      }

    For the Cobb measurement:
      - The TOP vertebra of the curve uses its SUPERIOR endplate
      - The BOTTOM vertebra of the curve uses its INFERIOR endplate
    """
    row = {"token": token}
    pca_data = {}
    MIN_VOXELS = 150

    # CTSpinoPelvic1K label scheme:
    #   1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 6=L6, 7=Sacrum,
    #   8=left_hip, 9=right_hip  (ignored — not in LUMBAR_LBLS)
    # Labels 8-9 are simply skipped by the loop below.
    # If max label is far outside this range (e.g. VerSe 20-26), warn.
    max_label = int(label_vol.max())
    if max_label > 10:
        warnings.warn(
            "Token "+str(token)+": max label is "+str(max_label)+
            ". Expected 1-9 for CTSpinoPelvic1K. May be VerSe-style labels "
            "(20-26) which need remapping before running this script."
        )

    for lid in LUMBAR_LBLS:
        vox = np.argwhere(label_vol == lid)
        if vox.shape[0] < MIN_VOXELS:
            continue
        world = voxels_to_world(vox.astype(np.float64), affine)
        body_centroid = world.mean(axis=0)

        # Compute BOTH endplates for every vertebra
        sup_proj, _, sup_ep = endplate_normal_coronal(world, affine, which="superior")
        inf_proj, _, inf_ep = endplate_normal_coronal(world, affine, which="inferior")

        pca_data[lid] = {
            "body_centroid": body_centroid,
            "sup": (sup_proj, sup_ep),
            "inf": (inf_proj, inf_ep),
        }

    row["labels_found"] = ",".join(LABEL_MAP[l] for l in sorted(pca_data.keys()))
    avail = sorted(pca_data.keys())

    if not avail:
        raise ValueError(
            "No lumbar vertebrae found (labels 1-7 all have < "+str(MIN_VOXELS)+
            " voxels). Check that the label file uses the CTSpinoPelvic1K "
            "scheme (1=L1 .. 7=Sacrum). VerSe labels (20-26) need remapping."
        )

    # ── Helper: Cobb angle between top vertebra's superior EP and
    #            bottom vertebra's correct EP ───────────────────────────────────
    def ep_for_bottom(lid):
        """
        Which endplate to use when a vertebra is the BOTTOM of the curve:
          - Sacrum (lid=7): SUPERIOR endplate = the S1 endplate (top face).
            The sacrum's inferior surface is the tip of the coccyx — far too
            low and anatomically wrong for Cobb measurement.
          - All other lumbar vertebrae: INFERIOR endplate (bottom face).
        """
        return "sup" if lid == 7 else "inf"

    def cobb_pair(top_lid, bot_lid):
        """
        Clinically correct Cobb angle:
          top vertebra -> superior endplate
          bot vertebra -> correct endplate per ep_for_bottom()
        """
        proj_top = pca_data[top_lid]["sup"][0]
        proj_bot = pca_data[bot_lid][ep_for_bottom(bot_lid)][0]
        return cobb_angle(proj_top, proj_bot)

    # ── Clinical Cobb pair selection ─────────────────────────────────────────
    #
    # Clinical practice (SRS standard):
    #   Radiologists DEFAULT to measuring the full lumbar span — L1 superior
    #   endplate to Sacrum (S1) superior endplate. They only deviate to a
    #   sub-span if a clearly distinct regional curve is present, which shows
    #   up as a substantially larger angle on a shorter segment.
    #
    # Algorithm:
    #   1. Compute the default full-span angle (best available from
    #      L1-Sacrum > L1-L5 > L1-L4 in priority order).
    #   2. Scan all non-adjacent pairs (min 2-label separation).
    #   3. Only use a non-default pair if it beats the default by > OVERRIDE_MARGIN.
    #      This prevents noisy short segments from overriding the clinically
    #      meaningful full-span measurement.
    #
    OVERRIDE_MARGIN = 3.0   # degrees — a sub-span must beat default by this much

    # Step 1: default full-span pair (highest priority first)
    default_pair_ids = (None, None)
    default_angle    = float("nan")
    for top_cand, bot_cand in [(1,7),(1,5),(1,4),(2,7),(2,5)]:
        if top_cand in pca_data and bot_cand in pca_data:
            default_pair_ids = (top_cand, bot_cand)
            default_angle    = cobb_pair(top_cand, bot_cand)
            break

    # Step 2: scan all non-adjacent pairs for a stronger regional curve
    MIN_SEP = 2
    best_alt_angle    = float("nan")
    best_alt_pair_ids = (None, None)
    for ii, top in enumerate(avail):
        for jj, bot in enumerate(avail):
            if jj <= ii + MIN_SEP - 1:
                continue
            if (top, bot) == default_pair_ids:
                continue   # already have this as default
            a = cobb_pair(top, bot)
            if math.isnan(best_alt_angle) or a > best_alt_angle:
                best_alt_angle    = a
                best_alt_pair_ids = (top, bot)

    # Step 3: use default unless a non-default pair is substantially larger
    if (not math.isnan(best_alt_angle)
            and not math.isnan(default_angle)
            and best_alt_angle > default_angle + OVERRIDE_MARGIN):
        chosen_ids   = best_alt_pair_ids
        chosen_angle = best_alt_angle
    elif not math.isnan(default_angle):
        chosen_ids   = default_pair_ids
        chosen_angle = default_angle
    else:
        chosen_ids   = best_alt_pair_ids
        chosen_angle = best_alt_angle

    top_id, bot_id = chosen_ids
    max_pair_ids   = chosen_ids
    max_cobb       = chosen_angle
    max_pair       = (LABEL_MAP.get(top_id,"?"), LABEL_MAP.get(bot_id,"?")) if top_id else ("","")

    row["true_cobb_deg"]  = round(max_cobb, 2) if not math.isnan(max_cobb) else ""
    row["true_cobb_pair"] = max_pair[0]+"-"+max_pair[1]

    # Grade on the true (auto-detected) Cobb angle
    grade, desc = grade_scoliosis(row["true_cobb_deg"])
    row["scoliosis_grade"]       = grade
    row["scoliosis_description"] = desc

    # ── Max lateral centroid deviation from L1-Sacrum midline ────────────────
    if 1 in pca_data and 7 in pca_data:
        c_l1  = pca_data[1]["body_centroid"]
        c_sac = pca_data[7]["body_centroid"]
        midline = c_sac - c_l1
        ml = np.linalg.norm(midline)
        md, mdl = 0.0, ""
        if ml > 1.0:
            for lid in avail:
                d = pca_data[lid]["body_centroid"] - c_l1
                perp = d - (np.dot(d, midline) / ml**2) * midline
                dev = abs(perp[0])
                if dev > md:
                    md, mdl = dev, LABEL_MAP.get(lid, str(lid))
        row["max_coronal_deviation_mm"] = round(md, 2)
        row["max_deviation_at"] = mdl
    else:
        row["max_coronal_deviation_mm"] = ""
        row["max_deviation_at"] = ""

    return row, pca_data, max_pair_ids

# ═════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def visualise_case(label_vol, affine, pca_data, row, out_path, token, cobb_pair_ids=(None,None)):
    """
    Coronal view showing:
      - Coloured vertebral labels
      - Centroid dots + endplate normal arrows
      - Highlighted endplate lines for the TRUE Cobb pair (auto-detected)
      - L1 and L5 reference lines (dashed) for comparison
      - Angle annotation and clinical grade box
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatch
    from matplotlib.lines import Line2D

    inv = np.linalg.inv(affine)
    si_vox_ax, si_sign = get_si_axis(affine)

    R = affine[:3, :3]
    ap_vox_ax = int(np.argmax(np.abs(R[1, :])))
    rl_vox_ax = int(np.argmax(np.abs(R[0, :])))

    ap_positions = []
    for lid in [1,2,3,4,5]:
        if lid not in pca_data: continue
        c_vox = (inv @ np.append(pca_data[lid]["body_centroid"], 1))[:3]
        ap_positions.append(c_vox[ap_vox_ax])
    if not ap_positions:
        return
    cut_ap = int(np.clip(round(float(np.median(ap_positions))),
                         0, label_vol.shape[ap_vox_ax]-1))

    if ap_vox_ax == 0:   cor_raw = label_vol[cut_ap, :, :]
    elif ap_vox_ax == 1: cor_raw = label_vol[:, cut_ap, :]
    else:                cor_raw = label_vol[:, :, cut_ap]

    remaining = [a for a in [0,1,2] if a != ap_vox_ax]
    cor_img = cor_raw.T if remaining[0] == rl_vox_ax else cor_raw
    if si_sign < 0:
        cor_img = np.flipud(cor_img)

    rgba = np.zeros((*cor_img.shape, 4), dtype=np.float32)
    for lid, col in COLORS.items():
        r = int(col[1:3],16)/255
        g = int(col[3:5],16)/255
        b = int(col[5:7],16)/255
        rgba[cor_img == lid] = (r, g, b, 0.9)

    def world_to_display(world_pt):
        c_vox = (inv @ np.append(world_pt, 1))[:3]
        x = c_vox[rl_vox_ax]
        si_raw = c_vox[si_vox_ax]
        y = (label_vol.shape[si_vox_ax] - 1 - si_raw) if si_sign < 0 else si_raw
        return float(x), float(y)

    # ── Per-label SI bounds from the largest contiguous row-group on slice ──────
    # Problem with using full label extent:
    #   cor_img == lid finds ALL pixels including detached lower fragments on
    #   the slice. rows.min() then returns the bottom of a stray fragment, not
    #   the bottom of the main vertebral body — so the clamp is too loose.
    #
    # Fix: find the largest CONTIGUOUS group of rows containing that label,
    #   then use that group's min/max. A gap of >3 rows between occupied rows
    #   marks a new fragment. This mirrors the 3D connected-cluster logic but
    #   applied to 2D pixel rows on the coronal slice.
    def largest_row_group_bounds(lid, gap=3):
        """
        Find the Y bounds of the largest contiguous row-group for label lid
        on the current coronal slice. Ignores detached row-groups (fragments).

        Returns (y_min, y_max) of the largest group, or None if label absent.
        """
        rows = np.where(np.any(cor_img == lid, axis=1))[0]
        if len(rows) == 0:
            return None
        if len(rows) == 1:
            return (float(rows[0]), float(rows[0]))

        # Split into contiguous groups wherever the gap between consecutive
        # occupied rows exceeds `gap` pixels
        breaks   = np.where(np.diff(rows) > gap)[0] + 1
        starts   = np.concatenate([[0], breaks])
        ends     = np.concatenate([breaks, [len(rows)]])

        # Pick the largest group by number of rows
        sizes    = ends - starts
        best     = int(np.argmax(sizes))
        group    = rows[starts[best]:ends[best]]
        return (float(group.min()), float(group.max()))

    label_slice_bounds = {}
    for lid in LABEL_MAP:
        b = largest_row_group_bounds(lid)
        if b is not None:
            label_slice_bounds[lid] = b

    def clamp_y_to_label(y, lid):
        """Clamp display Y to within the largest contiguous SI extent of lid."""
        if lid not in label_slice_bounds:
            return y
        y_min, y_max = label_slice_bounds[lid]
        return float(np.clip(y, y_min, y_max))

    def draw_ep_line(ax, ep_centroid, proj_2d, color, lid=None,
                     lw=2.5, ls="-", ep_len=48, zorder=8, label=None):
        ecx, ecy = world_to_display(ep_centroid)
        # Clamp Y to the visible extent of the vertebra on this slice
        if lid is not None:
            ecy = clamp_y_to_label(ecy, lid)
        ep_dx = -proj_2d[1]
        ep_dy =  proj_2d[0] * (1.0 if si_sign > 0 else -1.0)
        n = math.sqrt(ep_dx**2 + ep_dy**2) + 1e-9
        ep_dx /= n; ep_dy /= n
        ax.plot([ecx - ep_dx*ep_len, ecx + ep_dx*ep_len],
                [ecy - ep_dy*ep_len, ecy + ep_dy*ep_len],
                ls, color=color, lw=lw, zorder=zorder,
                solid_capstyle="round", label=label)

    fig, ax = plt.subplots(1, 1, figsize=(9, 12), facecolor="black")

    true_co  = row.get("true_cobb_deg", "N/A")
    pair_str = row.get("true_cobb_pair", "?-?")
    grade    = row.get("scoliosis_grade", "N/A")
    desc     = row.get("scoliosis_description", "")

    ax.set_facecolor("black")
    ax.imshow(rgba, origin="lower", aspect="auto")
    ax.set_title(
        "Coronal View — Token "+str(token)+
        "\nCobb ("+str(pair_str)+") = "+str(true_co)+"°   |   Grade: "+str(grade),
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Left  ←→  Right", color="white", fontsize=10)
    ax.set_ylabel("Inferior  →  Superior", color="white", fontsize=10)
    ax.tick_params(colors="#555555")

    # ── Centroid dots + normal arrows for all vertebrae ──────────────────────
    arrow_scale = 28.0
    for lid, vd in pca_data.items():
        if lid not in COLORS: continue
        col = COLORS[lid]
        dx, dy = world_to_display(vd["body_centroid"])
        ax.scatter(dx, dy, color=col, s=90, zorder=6,
                   edgecolors="white", linewidths=0.8)
        ax.text(dx+4, dy, LABEL_MAP.get(lid,""),
                color=col, fontsize=9, fontweight="bold", va="center")

        # Arrow along superior endplate normal
        sup_proj, sup_ep = vd["sup"]
        ecx, ecy = world_to_display(sup_ep)
        adx = sup_proj[0] * arrow_scale
        ady = sup_proj[1] * arrow_scale * (1.0 if si_sign > 0 else -1.0)
        ax.annotate("", xy=(ecx+adx, ecy+ady), xytext=(ecx-adx, ecy-ady),
            arrowprops=dict(arrowstyle="-|>", color=col,
                            mutation_scale=10, lw=1.5), zorder=7)

    top_id, bot_id = cobb_pair_ids

    # ── Draw TRUE Cobb endplate lines (solid, bright, thick) ─────────────────
    # Top vertebra: superior endplate (red)
    # Bot vertebra: inferior endplate (blue)
    if top_id is not None and top_id in pca_data:
        sup_proj, sup_ep = pca_data[top_id]["sup"]
        draw_ep_line(ax, sup_ep, sup_proj, "#ff4444", lid=top_id,
                     lw=3.0, ls="-", ep_len=52,
                     label=LABEL_MAP[top_id]+" sup endplate (Cobb top)")
    if bot_id is not None and bot_id in pca_data:
        # Sacrum: use superior (S1) endplate — its inferior surface is the
        # coccyx tip, which is anatomically wrong for Cobb measurement
        bot_ep_key = "sup" if bot_id == 7 else "inf"
        bot_proj, bot_ep = pca_data[bot_id][bot_ep_key]
        draw_ep_line(ax, bot_ep, bot_proj, "#44aaff", lid=bot_id,
                     lw=3.0, ls="-", ep_len=52,
                     label=LABEL_MAP[bot_id]+" endplate (Cobb bot)")

    # No reference lines — true Cobb pair only
    show_l1_ref = False
    show_l5_ref = False

    # ── Dashed midline L1-Sacrum ──────────────────────────────────────────────
    if 1 in pca_data and 7 in pca_data:
        x1, y1 = world_to_display(pca_data[1]["body_centroid"])
        x7, y7 = world_to_display(pca_data[7]["body_centroid"])
        ax.plot([x1,x7],[y1,y7],"--",color="white",alpha=0.25,lw=1.0,zorder=3)

    # ── Cobb angle label ──────────────────────────────────────────────────────
    if top_id in pca_data and bot_id in pca_data and true_co not in ("N/A",""):
        xt, yt = world_to_display(pca_data[top_id]["body_centroid"])
        xb, yb = world_to_display(pca_data[bot_id]["body_centroid"])
        mx, my = (xt+xb)/2 + 14, (yt+yb)/2
        ax.annotate(str(true_co)+"°  ("+pair_str+")",
                    xy=(mx, my), fontsize=11, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#1a1a2e", edgecolor="white", alpha=0.85))

    # ── Grade box ─────────────────────────────────────────────────────────────
    grade_colors = {
        "Normal":"#2a9d8f","Mild":"#e9c46a","Moderate":"#f4a261",
        "Severe":"#e63946","Very Severe":"#9b1c1c","N/A":"#555555",
    }
    gc = grade_colors.get(grade, "#555555")
    ax.text(0.02, 0.02,
            "Grade: "+str(grade)+"\n"+str(desc),
            transform=ax.transAxes, color="white", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=gc, edgecolor="white", alpha=0.85),
            va="bottom")

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [mpatch.Patch(color=COLORS[l], label=LABEL_MAP[l])
               for l in COLORS if l in pca_data]
    handles += [
        Line2D([0],[0], color="#ff4444", lw=2.5, label="Cobb top endplate"),
        Line2D([0],[0], color="#44aaff", lw=2.5, label="Cobb bot endplate"),
    ]
    if show_l1_ref:
        handles.append(Line2D([0],[0], color="#ff9999", lw=1.5, ls="--", label="L1 ref"))
    if show_l5_ref:
        handles.append(Line2D([0],[0], color="#99ccff", lw=1.5, ls="--", label="L5 ref"))
    ax.legend(handles=handles, loc="upper left",
              facecolor="#111111", edgecolor="#444444",
              labelcolor="white", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print("    Saved: "+out_path.name)

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(rows):
    from collections import Counter
    print("")
    print("="*72)
    print("  CORONAL COBB ANGLE — CLINICAL RESULTS SUMMARY")
    print("="*72)
    print("")
    print("  Scoliosis grading: Scoliosis Research Society (SRS) thresholds")
    print("  Metric: Coronal Cobb angle L1–L5 (endplate normal PCA method)")
    print("")

    grade_counts = Counter(r.get("scoliosis_grade","N/A") for r in rows)
    total = len(rows)

    print("  {:<15} {:<6} {:<8} {:<38}".format("Grade","N","%","Clinical Meaning"))
    print("  "+"-"*68)
    for lo, hi, grade, desc in SCOLIOSIS_GRADES:
        n = grade_counts.get(grade, 0)
        pct = 100.0*n/total if total > 0 else 0
        print("  {:<15} {:<6} {:<8} {:<38}".format(
            grade, str(n), str(round(pct,1))+"%", desc))
    n_na = grade_counts.get("N/A",0)
    if n_na:
        print("  {:<15} {:<6}".format("N/A", str(n_na)))

    print("")
    print("  Per-patient results:")
    print("  {:<8} {:<12} {:<22} {:<14}".format(
        "Token","Config","True Cobb (pair)","Grade"))
    print("  "+"-"*60)
    for r in rows:
        pair  = str(r.get("true_cobb_pair",""))
        true_c = str(r.get("true_cobb_deg",""))
        print("  {:<8} {:<12} {:<22} {:<14}".format(
            str(r.get("token","")),
            str(r.get("config","")),
            true_c+"° ("+pair+")" if true_c else "N/A",
            str(r.get("scoliosis_grade",""))
        ))

    # Summary stats on true Cobb
    vals = []
    for r in rows:
        v = r.get("true_cobb_deg","")
        if v:
            try: vals.append(float(v))
            except: pass
    if vals:
        a = np.array(vals)
        print("")
        print("  True Cobb angle statistics (n="+str(len(a))+"):")
        print("    Mean:   "+str(round(float(a.mean()),2))+" deg")
        print("    Median: "+str(round(float(np.median(a)),2))+" deg")
        print("    SD:     "+str(round(float(a.std()),2))+" deg")
        print("    Min:    "+str(round(float(a.min()),2))+" deg")
        print("    Max:    "+str(round(float(a.max()),2))+" deg")

    print("="*72)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET LOADING
# ═════════════════════════════════════════════════════════════════════════════

def build_ds(args):
    try:
        from dataset_interface import CTSpinoPelvic1K
    except ImportError:
        sys.exit("ERROR: dataset_interface.py not found in this folder.")
    if args.hub:
        print("Connecting to HuggingFace Hub: "+args.hub)
        ds = CTSpinoPelvic1K.from_hub(repo_id=args.hub)
        print(ds.stats())
        print("Ensuring all NIfTI files are local...")
        for i, case in enumerate(ds.cases):
            print("  ["+str(i+1)+"/"+str(len(ds.cases))+"] "+str(case.token)
                  +" ...", end=" ", flush=True)
            try:
                case._ensure_local(); print("OK")
            except Exception as e:
                print("FAILED: "+str(e))
        print()
    else:
        print("Loading from local folder: "+args.root)
        ds = CTSpinoPelvic1K(args.root)
        print(ds.stats())
    return ds


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def process_one_case(args_tuple):
    """
    Worker function — runs in a separate process.
    Each worker loads one case, runs analysis, saves PNG, returns the result row.
    Must be a top-level function (not a closure) for multiprocessing to pickle it.

    Returns (row_dict, token) on success, or (None, token) on failure.
    """
    case, out_dir_str, no_viz, verbose = args_tuple
    out_dir = Path(out_dir_str)
    token = case.token

    try:
        label_vol, affine = case.load_label()
    except Exception as e:
        return None, token, str(e)

    try:
        row, pca_data, cobb_pair_ids = analyse_case(label_vol, affine, token)
        row["config"] = case.config
    except Exception as e:
        return None, token, str(e)

    if not no_viz:
        vp = out_dir / ("cobb_viz_"+str(token)+".png")
        try:
            visualise_case(label_vol, affine, pca_data, row, vp,
                           token, cobb_pair_ids=cobb_pair_ids)
        except Exception as e:
            pass  # viz failure doesn't fail the whole case

    return row, token, None


def main():
    p = argparse.ArgumentParser(
        description="Coronal Cobb angle from CTSpinoPelvic1K segmentation masks.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--root",  metavar="DIR",     help="Local dataset folder")
    g.add_argument("--hub",   metavar="REPO_ID", help="HuggingFace Hub repo ID")
    p.add_argument("--configs", default="fused,spine_only",
                   help="Comma-separated configs to include (default: fused,spine_only)")
    p.add_argument("--output-csv", default="cobb_angles.csv")
    p.add_argument("--output-dir", default=".",
                   help="Folder for PNG images")
    p.add_argument("--no-viz", action="store_true",
                   help="Skip visualisation (faster for large batches)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers (default: number of CPUs)")
    p.add_argument("--no-progress", action="store_true",
                   help="Disable tqdm progress bar")
    args = p.parse_args()

    # ── Logging setup ────────────────────────────────────────────────────────
    log_path = Path(args.output_dir) / "cobb_analysis.log"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger("cobb")
    log.info("Starting Cobb angle analysis")
    log.info("Output dir: "+str(Path(args.output_dir).resolve()))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_ds(args)

    # Collect cases for all requested configs
    configs = [c.strip() for c in args.configs.split(",")]
    cases = []
    for cfg in configs:
        found = ds.filter(config=cfg, present_only=True)
        cases += found
        log.info("Config '"+cfg+"': "+str(len(found))+" cases present on disk")

    if not cases:
        cases = [c for c in ds.cases if c.exists()]
        print("Fallback: using all "+str(len(cases))+" present cases")

    if not cases:
        sys.exit("ERROR: No cases found on disk.")

    log.info(f"\nTotal cases to process: {len(cases)}\n")

    cols = [
        "token", "config", "labels_found",
        "true_cobb_deg", "true_cobb_pair",
        "max_coronal_deviation_mm", "max_deviation_at",
        "scoliosis_grade", "scoliosis_description",
    ]

    csv_path = Path(args.output_csv)
    all_rows = []
    n_ok = 0
    n_skip = 0

    # ── Thread-safe CSV writer ────────────────────────────────────────────────
    csv_lock = threading.Lock()

    import os
    n_workers = args.workers or min(48, os.cpu_count() or 4)
    log.info(f"Using {n_workers} parallel workers for {len(cases)} cases")

    # Build argument tuples for workers
    work_items = [
        (case, str(out_dir), args.no_viz, args.verbose)
        for case in cases
    ]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        fh.flush()

        # ── Progress bar ──────────────────────────────────────────────────────
        pbar = tqdm(
            total=len(cases),
            desc="Analysing",
            unit="case",
            disable=args.no_progress,
            dynamic_ncols=True,
        )

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_one_case, item): item[0]
                       for item in work_items}

            for future in as_completed(futures):
                case = futures[future]
                try:
                    row, token, err = future.result()
                except Exception as e:
                    log.error(f"  Worker crash token={case.token}: {e}")
                    n_skip += 1
                    pbar.update(1)
                    continue

                if row is None:
                    log.warning(f"  Skipped token={token}: {err}")
                    n_skip += 1
                else:
                    # Thread-safe CSV write + flush
                    with csv_lock:
                        writer.writerow(row)
                        fh.flush()
                    all_rows.append(row)
                    n_ok += 1
                    cobb = row.get("true_cobb_deg", "N/A")
                    pair = row.get("true_cobb_pair", "?")
                    grade = row.get("scoliosis_grade", "N/A")
                    log.info(f"  token={token} Cobb({pair})={cobb}° Grade={grade}")

                pbar.update(1)
                pbar.set_postfix(ok=n_ok, skip=n_skip)

        pbar.close()

    log.info("")
    log.info(f"Processed {n_ok} cases successfully, {n_skip} skipped.")
    log.info(f"CSV: {csv_path.resolve()}")
    log.info(f"Log: {log_path.resolve()}")

    if all_rows:
        print_summary(all_rows)

    log.info("")
    log.info(f"Open CSV:  open \"{csv_path.resolve()}\"")


if __name__ == "__main__":
    main()
