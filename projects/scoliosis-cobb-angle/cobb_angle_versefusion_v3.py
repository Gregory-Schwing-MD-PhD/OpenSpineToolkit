"""
cobb_angle_versefusion_v3.py
============================
Automated thoracic AND lumbar Cobb angle measurement on the VerseFusion dataset.

KEY IMPROVEMENTS OVER v2:
  1. ITERATIVE PLANE-NORMAL ENDPLATE ISOLATION (replaces naive SI-coordinate
     slicing). Old method: take top/bottom 15% of voxels by world-Z coordinate.
     Problem: for a rotated vertebra the "top 15% by Z" grabs the wrong surface —
     it clips a corner rather than the true endplate.

     New method (isolate_endplate_v3):
       Pass 1 — rough estimate: take top/bottom 20% by SI coord → fit a plane.
       Pass 2 — project ALL vertebral voxels onto that plane normal, take the
                 top/bottom 15% by projection distance → refit the plane.
       Pass 3 — repeat once more for stability.
     This converges to the true endplate surface regardless of vertebral rotation,
     which is exactly what breaks down in scoliotic spines.

  2. MIN SPAN ENFORCEMENT (from v2) — thoracic pair must span ≥ MIN_THORACIC_SPAN
     label indices (default 4), eliminating T10-T12 / T11-T12 spurious pairs.

  3. OUTLIER REJECTION per pass via IQR on projection coordinate, removing
     partial-volume edge voxels that corrupt the plane fit.

  4. Comparison figure generator built in (--compare mode).
     Outputs CT + segmentation overlay, 3-panel (Normal / Lumbar / Thoracic).

Usage:
    # Full analysis run:
    python cobb_angle_versefusion_v3.py \\
        --data-dir   ~/cobb/versefusion_data/ \\
        --output-dir ~/cobb/versefusion_results_v3/ \\
        --workers    48

    # Comparison figure only (after analysis):
    python cobb_angle_versefusion_v3.py \\
        --compare \\
        --csv  ~/cobb/versefusion_results_v3/cobb_versefusion_v3.csv \\
        --data ~/cobb/versefusion_data/scans/ \\
        --out  ~/cobb/versefusion_figures_v3/
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patheffects import withStroke
import numpy as np
from tqdm import tqdm


# ── Label schema ──────────────────────────────────────────────────────────────

VERSE_LABEL_MAP = {
    1:"C1",  2:"C2",  3:"C3",  4:"C4",  5:"C5",  6:"C6",  7:"C7",
    8:"T1",  9:"T2",  10:"T3", 11:"T4", 12:"T5", 13:"T6",
    14:"T7", 15:"T8", 16:"T9", 17:"T10",18:"T11",19:"T12",
    20:"L1", 21:"L2", 22:"L3", 23:"L4", 24:"L5", 25:"L6",
    26:"Sacrum", 27:"Coccyx", 28:"T13",
}
NAME_TO_LID = {v: k for k, v in VERSE_LABEL_MAP.items()}

THORACIC_LBLS  = list(range(11, 20))   # T4(11) .. T12(19)
LUMBAR_LBLS    = list(range(20, 27))   # L1(20) .. Sacrum(26)
ALL_SPINE_LBLS = list(range(1,  27))

# Minimum vertebral span for a valid Cobb pair
MIN_THORACIC_SPAN = 4   # e.g. T4-T8 is span 4; T10-T12 is span 2 → rejected
MIN_LUMBAR_SPAN   = 2   # L1-L2 is the smallest accepted lumbar pair

MIN_VOXELS = 150        # reject a vertebra with fewer world-voxels than this

# Visualization colors
COLORS = {
    **{i: "#9b59b6" for i in range(1, 8)},     # Cervical: purple
    **{i: "#3498db" for i in range(8, 20)},     # Thoracic: steel blue
    20:"#e63946", 21:"#f4a261", 22:"#2a9d8f",   # L1-L3
    23:"#457b9d", 24:"#a8dadc", 25:"#6d6875",   # L4-L6
    26:"#e9c46a",                                # Sacrum: gold
}


def srs_grade(angle: float) -> str:
    if angle < 10:  return "Normal"
    if angle < 25:  return "Mild"
    if angle < 40:  return "Moderate"
    if angle < 60:  return "Severe"
    return "Very Severe"


# ═══════════════════════════════════════════════════════════════════════════════
# CORE GEOMETRY — v3 endplate isolation
# ═══════════════════════════════════════════════════════════════════════════════

def vox2world(vox: np.ndarray, aff: np.ndarray) -> np.ndarray:
    """Convert Nx3 voxel coordinates to world (mm) coordinates."""
    h = np.ones((len(vox), 4))
    h[:, :3] = vox
    return (aff @ h.T).T[:, :3]


def largest_connected_cluster(pts: np.ndarray, gap_mm: float = 8.0) -> np.ndarray:
    """Return the largest gap-connected cluster of 3D points along SI axis."""
    if len(pts) == 0:
        return pts
    order    = np.argsort(pts[:, 2])
    sorted_z = pts[order, 2]
    gaps     = np.diff(sorted_z)
    breaks   = np.where(gaps > gap_mm)[0] + 1
    starts   = np.concatenate([[0], breaks])
    ends     = np.concatenate([breaks, [len(pts)]])
    best     = int(np.argmax(ends - starts))
    return pts[order[starts[best]:ends[best]]]


def fit_plane_pca(pts: np.ndarray):
    """
    Fit a plane through pts via PCA (smallest singular value = normal).
    Returns (centroid, normal) where normal points in +SI direction.
    """
    c       = pts.mean(0)
    _, s, Vt = np.linalg.svd(pts - c, full_matrices=False)
    n       = Vt[np.argmin(s)].copy()
    if n[2] < 0:
        n = -n      # ensure normal points superior
    return c, n


def iqr_filter(vals: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Return boolean mask of values within k*IQR of median."""
    q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
    iqr    = q3 - q1
    if iqr < 1e-6:
        return np.ones(len(vals), dtype=bool)
    return (vals >= q1 - k*iqr) & (vals <= q3 + k*iqr)


def isolate_endplate_v3(world_pts: np.ndarray,
                        which: str = "superior",
                        frac: float = 0.15,
                        min_pts: int = 40,
                        n_iter: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative plane-normal endplate isolation.

    Algorithm
    ---------
    Instead of slicing by world-Z (which fails for rotated vertebrae),
    we iteratively project all vertebral voxels onto the current best-estimate
    plane normal and select the extreme fraction along that direction.

    Pass 0  — seed: select top/bottom 20% by raw SI (world-Z) coordinate.
              Fit a plane → get initial normal estimate.
    Pass 1..n_iter — project ALL cleaned voxels onto current normal.
              Select top/bottom `frac` fraction by projection → refit plane.
              Apply IQR outlier rejection on projection values each pass.

    Parameters
    ----------
    world_pts : (N,3) array of vertebral voxels in world (mm) coords
    which     : "superior" or "inferior"
    frac      : fraction of voxels to keep as endplate surface (default 0.15)
    min_pts   : minimum voxel count to attempt isolation
    n_iter    : number of refinement iterations (default 3)

    Returns
    -------
    ep_pts    : (M,3) endplate surface points
    ep_centroid : (3,) centroid of endplate surface
    normal    : (3,) unit normal of fitted plane (points superior)
    """
    # ── Seed from largest cluster ─────────────────────────────────────────────
    if len(world_pts) < min_pts:
        ep_c, norm = fit_plane_pca(world_pts)
        return world_pts, ep_c, norm

    main = largest_connected_cluster(world_pts)
    if len(main) < min_pts:
        main = world_pts

    # ── Pass 0: crude SI-coordinate seed ─────────────────────────────────────
    si_vals = main[:, 2]
    # IQR filter first to remove gross outliers
    mask0   = iqr_filter(si_vals, k=2.0)
    cleaned = main[mask0] if mask0.sum() >= min_pts else main

    n_seed  = max(min_pts, int(len(cleaned) * 0.20))   # 20% for seed
    si_c    = cleaned[:, 2]
    if which == "superior":
        seed_idx = np.argpartition(si_c, -n_seed)[-n_seed:]
    else:
        seed_idx = np.argpartition(si_c,  n_seed)[:n_seed]
    seed_pts = cleaned[seed_idx]

    if len(seed_pts) < 3:
        ep_c, norm = fit_plane_pca(cleaned)
        return seed_pts, ep_c, norm

    _, norm = fit_plane_pca(seed_pts)

    # ── Passes 1..n_iter: project onto normal, reselect, refit ───────────────
    work = cleaned
    for _ in range(n_iter):
        # Project all voxels onto current normal
        proj    = work @ norm          # signed distance along normal direction

        # IQR filter on projection to remove partial-volume edge voxels
        pmask   = iqr_filter(proj, k=1.5)
        work_f  = work[pmask] if pmask.sum() >= min_pts else work
        proj_f  = (work_f @ norm)

        # Select extreme fraction
        n_take  = max(min_pts, int(len(work_f) * frac))
        n_take  = min(n_take, len(work_f))
        if which == "superior":
            idx = np.argpartition(proj_f, -n_take)[-n_take:]
        else:
            idx = np.argpartition(proj_f,  n_take)[:n_take]

        ep_pts = work_f[idx]
        if len(ep_pts) < 3:
            break
        _, norm_new = fit_plane_pca(ep_pts)
        # Ensure sign consistency
        if np.dot(norm_new, norm) < 0:
            norm_new = -norm_new
        norm = norm_new

    ep_c, _ = fit_plane_pca(ep_pts)
    return ep_pts, ep_c, norm


def ep_normal_coronal_v3(world_pts: np.ndarray,
                         which: str = "superior") -> tuple:
    """
    Compute endplate normal using v3 iterative isolation.

    Returns
    -------
    proj_2d      : (2,) unit vector in coronal plane [RL, SI]
    body_centroid: (3,) mean of all vertebral voxels
    ep_centroid  : (3,) centroid of isolated endplate surface
    normal_3d    : (3,) full 3D plane normal
    """
    body_c          = world_pts.mean(0)
    ep_pts, ep_c, norm = isolate_endplate_v3(world_pts, which)

    # Project 3D normal into coronal plane (ignore AP component)
    proj = np.array([norm[0], norm[2]])   # [RL, SI]
    nm   = np.linalg.norm(proj)
    proj = proj / nm if nm > 1e-9 else np.array([0., 1.])

    return proj, body_c, ep_c, norm


def cobb_from_normals(p1: np.ndarray, p2: np.ndarray) -> float:
    cos_a     = float(np.clip(np.dot(p1, p2), -1, 1))
    angle_deg = math.degrees(math.acos(cos_a))
    return min(angle_deg, 180 - angle_deg)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CASE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def build_pca_data(label_vol: np.ndarray, affine: np.ndarray,
                   label_ids: list) -> dict:
    """
    For each vertebra label: compute superior and inferior endplate normals
    using v3 iterative isolation. Returns dict keyed by label_id.
    """
    pca = {}
    for lid in label_ids:
        vox = np.argwhere(label_vol == lid)
        if len(vox) < MIN_VOXELS:
            continue
        world = vox2world(vox.astype(np.float64), affine)
        pca[lid] = {
            "sup":   ep_normal_coronal_v3(world, "superior"),
            "inf":   ep_normal_coronal_v3(world, "inferior"),
            "world": world,
        }
    return pca


def find_best_pair(pca: dict, label_ids: list,
                   min_span: int = 1) -> tuple:
    """
    Find the vertebral pair giving the maximum Cobb angle.
    Pairs with label-index difference < min_span are skipped (span gate).

    Returns (top_id, bot_id, angle_deg, top_proj, bot_proj).
    """
    avail = sorted([l for l in label_ids if l in pca])
    if len(avail) < 2:
        return None, None, 0.0, None, None

    best = (None, None, 0.0, None, None)
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            top_id, bot_id = avail[i], avail[j]
            if (bot_id - top_id) < min_span:
                continue
            tp    = pca[top_id]["sup"][0]
            bp    = pca[bot_id]["sup"][0] if bot_id == 26 \
                    else pca[bot_id]["inf"][0]
            angle = cobb_from_normals(tp, bp)
            if angle > best[2]:
                best = (top_id, bot_id, angle, tp, bp)

    return best


def analyse_case(label_vol: np.ndarray, affine: np.ndarray,
                 case_id: str) -> tuple[dict, dict]:
    """Full thoracic + lumbar Cobb analysis. Returns (row_dict, pca_data)."""
    row   = {"case_id": case_id}
    avail = [l for l in ALL_SPINE_LBLS if l <= int(label_vol.max())]
    pca   = build_pca_data(label_vol, affine, avail)

    row["labels_found"] = ",".join(VERSE_LABEL_MAP.get(l, str(l))
                                   for l in sorted(pca))
    row["n_vertebrae"]  = len(pca)

    # ── Lumbar Cobb ───────────────────────────────────────────────────────────
    top, bot, ang, tp, bp = find_best_pair(pca, LUMBAR_LBLS,
                                           min_span=MIN_LUMBAR_SPAN)
    if top is not None:
        row["lumbar_cobb_deg"]  = round(ang, 2)
        row["lumbar_cobb_pair"] = (f"{VERSE_LABEL_MAP[top]}-"
                                   f"{VERSE_LABEL_MAP[bot]}")
        row["lumbar_grade"]     = srs_grade(ang)
        if 20 in pca and 26 in pca:
            row["lumbar_L1_S1_ref"] = round(cobb_from_normals(
                pca[20]["sup"][0], pca[26]["sup"][0]), 2)
        if 20 in pca and 24 in pca:
            row["lumbar_L1_L5_ref"] = round(cobb_from_normals(
                pca[20]["sup"][0], pca[24]["inf"][0]), 2)
    else:
        row["lumbar_cobb_deg"] = row["lumbar_cobb_pair"] = \
        row["lumbar_grade"]    = None

    # ── Thoracic Cobb (T4-T12) ────────────────────────────────────────────────
    top, bot, ang, tp, bp = find_best_pair(pca, THORACIC_LBLS,
                                           min_span=MIN_THORACIC_SPAN)
    if top is not None:
        row["thoracic_cobb_deg"]  = round(ang, 2)
        row["thoracic_cobb_pair"] = (f"{VERSE_LABEL_MAP[top]}-"
                                     f"{VERSE_LABEL_MAP[bot]}")
        row["thoracic_grade"]     = srs_grade(ang)
    else:
        row["thoracic_cobb_deg"] = row["thoracic_cobb_pair"] = \
        row["thoracic_grade"]    = None

    # Fixed T4-T12 anatomic reference (always, if both present)
    if 11 in pca and 19 in pca:
        row["thoracic_T4_T12_ref"] = round(cobb_from_normals(
            pca[11]["sup"][0], pca[19]["inf"][0]), 2)

    # ── Combined best pair ────────────────────────────────────────────────────
    top, bot, ang, tp, bp = find_best_pair(pca, sorted(pca.keys()), min_span=1)
    if top is not None:
        row["combined_cobb_deg"]  = round(ang, 2)
        row["combined_cobb_pair"] = (f"{VERSE_LABEL_MAP.get(top,'?')}-"
                                     f"{VERSE_LABEL_MAP.get(bot,'?')}")
        row["combined_grade"]     = srs_grade(ang)

    return row, pca


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_display_axes(affine: np.ndarray):
    """Return (ap_ax, rl_ax, si_ax, si_sgn, rl_sgn) from affine."""
    R      = affine[:3, :3]
    ap_ax  = int(np.argmax(np.abs(R[1, :])))
    rl_ax  = int(np.argmax(np.abs(R[0, :])))
    si_ax  = int(np.argmax(np.abs(R[2, :])))
    si_sgn = float(np.sign(R[2, si_ax]))
    rl_sgn = float(np.sign(R[0, rl_ax]))
    return ap_ax, rl_ax, si_ax, si_sgn, rl_sgn


def coronal_slice(vol: np.ndarray, affine: np.ndarray,
                  ap_ax: int, rl_ax: int, si_sgn: float,
                  ap_cut: int = None) -> np.ndarray:
    """
    Extract coronal slice at ap_cut, oriented inferior→superior (rows),
    left→right (cols).
    """
    if ap_cut is None:
        ap_cut = vol.shape[ap_ax] // 2
    slc        = [slice(None)] * 3
    slc[ap_ax] = ap_cut
    sl         = vol[tuple(slc)]
    remaining  = [a for a in [0, 1, 2] if a != ap_ax]
    if remaining[0] == rl_ax:
        sl = sl.T
    if si_sgn < 0:
        sl = np.flipud(sl)
    return sl


def ep_display_coords(pca: dict, lid: int, which: str,
                      affine: np.ndarray,
                      ap_ax: int, rl_ax: int, si_ax: int,
                      si_sgn: float, lbl_sl: np.ndarray):
    """
    Convert 3D endplate centroid → 2D display coordinates (col, row).
    Uses the v3 endplate centroid (ep_c = pca[lid][which][2]).
    Clamps row to the vertebra's visible pixel range in the 2D slice.
    """
    inv    = np.linalg.inv(affine)
    ep_c   = pca[lid][which][2]                     # 3D world centroid
    c_vox  = (inv @ np.append(ep_c, 1.0))[:3]
    ecx    = float(c_vox[rl_ax])
    si_raw = float(c_vox[si_ax])
    si_sz  = lbl_sl.shape[0]
    ecy    = (si_sz - 1 - si_raw) if si_sgn < 0 else si_raw

    rows_lid = np.where(np.any(lbl_sl == lid, axis=1))[0]
    if len(rows_lid) > 0:
        ecy = float(np.clip(ecy, rows_lid.min(), rows_lid.max()))
    return ecx, ecy


def draw_ep_line(ax, pca: dict, lid: int, which: str,
                 color: str, affine: np.ndarray,
                 ap_ax: int, rl_ax: int, si_ax: int,
                 si_sgn: float, lbl_sl: np.ndarray,
                 ep_len: float = 60, lw: float = 2.8):
    """
    Draw an endplate line on ax.

    Direction: perpendicular to the v3 plane normal projected onto the coronal
    display plane. Anchor: v3 endplate centroid reprojected to 2D.
    """
    if lid not in pca:
        return
    proj_2d = pca[lid][which][0]        # (RL, SI) unit vector from v3
    ecx, ecy = ep_display_coords(pca, lid, which, affine,
                                 ap_ax, rl_ax, si_ax, si_sgn, lbl_sl)

    # The line direction is perpendicular to the normal in display space.
    # normal in display = [proj_2d[0]=RL, proj_2d[1]=SI]
    # perpendicular (endplate direction) = [-SI, RL] corrected for SI flip
    si_dir = 1.0 if si_sgn > 0 else -1.0
    dx     = -proj_2d[1] * si_dir       # endplate runs in RL direction
    dy     =  proj_2d[0]
    nm     = math.sqrt(dx**2 + dy**2) + 1e-9
    dx    /= nm
    dy    /= nm

    ax.plot([ecx - dx*ep_len, ecx + dx*ep_len],
            [ecy - dy*ep_len, ecy + dy*ep_len],
            "-", color=color, lw=lw, zorder=9, solid_capstyle="round",
            path_effects=[withStroke(linewidth=lw + 2.0,
                                     foreground="black", alpha=0.55)])


def build_ct_rgba(ct_sl: np.ndarray, lbl_sl: np.ndarray,
                  alpha: float = 0.38,
                  wl: float = 50, ww: float = 400) -> np.ndarray:
    """CT window → grayscale RGBA with semi-transparent segmentation overlay."""
    ct_n = np.clip((ct_sl - (wl - ww/2)) / ww, 0, 1)
    rgba = np.stack([ct_n]*3 + [np.ones_like(ct_n)], axis=-1).astype(np.float32)
    for lid, col in COLORS.items():
        mask = lbl_sl == lid
        if not mask.any():
            continue
        r = int(col[1:3], 16) / 255.
        g = int(col[3:5], 16) / 255.
        b = int(col[5:7], 16) / 255.
        rgba[mask, 0] = rgba[mask, 0] * (1-alpha) + r * alpha
        rgba[mask, 1] = rgba[mask, 1] * (1-alpha) + g * alpha
        rgba[mask, 2] = rgba[mask, 2] * (1-alpha) + b * alpha
    return rgba


def spine_crop(lbl_sl: np.ndarray, pad_frac: float = 0.07):
    """Return (r0, r1) row crop that fits the spine with padding."""
    rows_s = np.where(np.any(np.isin(lbl_sl, ALL_SPINE_LBLS), axis=1))[0]
    if len(rows_s) == 0:
        return 0, lbl_sl.shape[0]
    pad = int(lbl_sl.shape[0] * pad_frac)
    return max(0, rows_s.min()-pad), min(lbl_sl.shape[0], rows_s.max()+pad)


def add_legend(ax, lbl_sl):
    handles = []
    for lid, name in [(11,"T4"),(19,"T12"),(20,"L1"),(24,"L5"),(26,"Sacrum")]:
        if lbl_sl is not None and not (lbl_sl == lid).any():
            continue
        handles.append(mpatch.Patch(color=COLORS.get(lid,"#888"),
                                    label=name, alpha=0.88))
    handles += [
        plt.Line2D([0],[0], color="#ff4444", lw=2.5, label="Lumbar top EP"),
        plt.Line2D([0],[0], color="#44aaff", lw=2.5, label="Lumbar bot EP"),
        plt.Line2D([0],[0], color="#ff9900", lw=2.5, label="Thoracic top EP"),
        plt.Line2D([0],[0], color="#00ccff", lw=2.5, label="Thoracic bot EP"),
    ]
    ax.legend(handles=handles, loc="upper left",
              facecolor="#111111", edgecolor="#555555",
              labelcolor="white", fontsize=7.5, framealpha=0.92)


def draw_pairs(ax, row, pca, affine, ap_ax, rl_ax, si_ax, si_sgn, lbl_sl,
               ep_len=60):
    """Draw lumbar and thoracic endplate line pairs."""
    for pair_key, top_col, bot_col in [
            ("lumbar_cobb_pair",   "#ff4444", "#44aaff"),
            ("thoracic_cobb_pair", "#ff9900", "#00ccff")]:
        pair = str(row.get(pair_key, "") or "")
        if not pair or "-" not in pair:
            continue
        parts   = pair.split("-")
        top_lid = NAME_TO_LID.get(parts[0])
        bot_lid = NAME_TO_LID.get(parts[1])
        if top_lid:
            draw_ep_line(ax, pca, top_lid, "sup", top_col,
                         affine, ap_ax, rl_ax, si_ax, si_sgn, lbl_sl, ep_len)
        if bot_lid:
            w = "sup" if bot_lid == 26 else "inf"
            draw_ep_line(ax, pca, bot_lid, w, bot_col,
                         affine, ap_ax, rl_ax, si_ax, si_sgn, lbl_sl, ep_len)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-CASE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_case(label_vol, ct_vol, affine, row, pca, out_path):
    ap_ax, rl_ax, si_ax, si_sgn, _ = get_display_axes(affine)

    ap_pos = [float(np.argwhere(label_vol==l).mean(0)[ap_ax])
              for l in pca if len(np.argwhere(label_vol==l)) > 0]
    cut    = int(np.clip(round(float(np.median(ap_pos))),
                         0, label_vol.shape[ap_ax]-1)) if ap_pos \
             else label_vol.shape[ap_ax] // 2

    ct_sl  = coronal_slice(ct_vol,    affine, ap_ax, rl_ax, si_sgn, cut).astype(np.float32)
    lbl_sl = coronal_slice(label_vol, affine, ap_ax, rl_ax, si_sgn, cut)

    rgba   = build_ct_rgba(ct_sl, lbl_sl)
    r0, r1 = spine_crop(lbl_sl)

    fig, ax = plt.subplots(figsize=(6, 12), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(rgba[r0:r1], origin="lower", aspect="auto",
              extent=[0, lbl_sl.shape[1], r0, r1])
    ax.set_ylim(r0, r1)

    draw_pairs(ax, row, pca, affine, ap_ax, rl_ax, si_ax, si_sgn, lbl_sl)

    lf = row.get("labels_found","")
    lf_parts = lf.split(",") if lf else ["?","?"]
    ax.set_title(
        f"Case {row['case_id']}  [{lf_parts[0]}–{lf_parts[-1]}]\n"
        f"Lumbar: {row.get('lumbar_cobb_deg','N/A')}°"
        f" ({row.get('lumbar_cobb_pair','?')})  |"
        f"  Thoracic: {row.get('thoracic_cobb_deg','N/A')}°"
        f" ({row.get('thoracic_cobb_pair','?')})",
        color="white", fontsize=9, fontweight="bold", pad=5)
    ax.set_xlabel("Left ←→ Right",      color="white", fontsize=9)
    ax.set_ylabel("Inferior → Superior", color="white", fontsize=9)
    ax.tick_params(colors="#555555")
    add_legend(ax, lbl_sl)

    for pair_key, grade_key, deg_key, ypos, gc in [
            ("lumbar_cobb_pair",   "lumbar_grade",   "lumbar_cobb_deg",   0.12, "#e9c46a"),
            ("thoracic_cobb_pair", "thoracic_grade",  "thoracic_cobb_deg", 0.05, "#2e86c1")]:
        if row.get(pair_key):
            ax.text(0.02, ypos,
                    f"{pair_key.split('_')[0].title()}: "
                    f"{row.get(deg_key,'?')}° ({row.get(pair_key,'?')})\n"
                    f"{row.get(grade_key,'?')} scoliosis",
                    transform=ax.transAxes, color="white", fontsize=7.5,
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=gc,
                              edgecolor="white", alpha=0.85))

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 3-PANEL COMPARISON FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CASES = {
    "normal":   "verse765",   # L=0.88°, T=2.90°,  full T1-L4
    "lumbar":   "verse047",   # L=10.44°, T=2.62°, full T1-L5
    "thoracic": "verse264",   # T=12.79° T4-T12,   full T1-L5
}
PANEL_COLORS = {"normal":"#27ae60", "lumbar":"#e9c46a", "thoracic":"#2e86c1"}
PANEL_TITLES = {"normal":"Normal Spine", "lumbar":"Lumbar Scoliosis",
                "thoracic":"Thoracic Scoliosis"}


def _load_case(scans_dir: Path, case_id: str):
    import nibabel as nib
    ct_path  = scans_dir / case_id / "ct.nii.gz"
    msk_path = scans_dir / case_id / "mask.nii.gz"
    if not ct_path.exists() or not msk_path.exists():
        return None, None, None
    ct_img  = nib.load(str(ct_path))
    msk_img = nib.load(str(msk_path))
    ct_vol  = np.asarray(ct_img.get_fdata(),  dtype=np.float32)
    lbl_vol = np.asarray(msk_img.get_fdata(), dtype=np.uint8)
    return ct_vol, lbl_vol, msk_img.affine.astype(np.float64)


def draw_comparison_panel(ax, case_id, row, scans_dir):
    ct_vol, lbl_vol, affine = _load_case(scans_dir, case_id)
    if ct_vol is None:
        ax.set_facecolor("black")
        ax.text(0.5, 0.5, f"Missing\n{case_id}", ha="center", va="center",
                color="white", transform=ax.transAxes, fontsize=10)
        return None, None, None

    # Re-run v3 analysis on this case
    _, pca = analyse_case(lbl_vol, affine, case_id)

    ap_ax, rl_ax, si_ax, si_sgn, _ = get_display_axes(affine)
    ap_pos = [float(np.argwhere(lbl_vol==l).mean(0)[ap_ax])
              for l in pca if len(np.argwhere(lbl_vol==l)) > 0]
    cut    = int(np.clip(round(float(np.median(ap_pos))),
                         0, lbl_vol.shape[ap_ax]-1)) if ap_pos \
             else lbl_vol.shape[ap_ax] // 2

    ct_sl  = coronal_slice(ct_vol,  affine, ap_ax, rl_ax, si_sgn, cut).astype(np.float32)
    lbl_sl = coronal_slice(lbl_vol, affine, ap_ax, rl_ax, si_sgn, cut)
    rgba   = build_ct_rgba(ct_sl, lbl_sl, alpha=0.38)
    r0, r1 = spine_crop(lbl_sl, pad_frac=0.06)

    ax.set_facecolor("black")
    ax.imshow(rgba[r0:r1], origin="lower", aspect="auto",
              extent=[0, lbl_sl.shape[1], r0, r1])
    ax.set_ylim(r0, r1)

    draw_pairs(ax, row, pca, affine, ap_ax, rl_ax, si_ax, si_sgn, lbl_sl,
               ep_len=65)
    ax.set_xlabel("Left ←→ Right",      color="white", fontsize=9)
    ax.set_ylabel("Inferior → Superior", color="white", fontsize=9)
    ax.tick_params(colors="#555555")
    return lbl_sl, pca, affine


def make_comparison_figure(csv_path, scans_dir, out_dir,
                           case_normal, case_lumbar, case_thoracic):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ["lumbar_cobb_deg", "thoracic_cobb_deg"]:
                try:    r[k] = float(r[k]) if r[k] else None
                except: r[k] = None
            rows[r["case_id"]] = r

    fig, axes = plt.subplots(1, 3, figsize=(20, 16), facecolor="black")
    fig.subplots_adjust(wspace=0.08)

    panels = [(axes[0], case_normal,   "normal"),
              (axes[1], case_lumbar,   "lumbar"),
              (axes[2], case_thoracic, "thoracic")]

    for ax, case_id, cat in panels:
        row    = rows.get(case_id, {})
        result = draw_comparison_panel(ax, case_id, row, scans_dir)
        lbl_sl = result[0] if result else None

        lumb   = row.get("lumbar_cobb_deg",   "N/A")
        lp     = row.get("lumbar_cobb_pair",  "?")
        lg     = row.get("lumbar_grade",      "?")
        thor   = row.get("thoracic_cobb_deg", "N/A")
        tp     = row.get("thoracic_cobb_pair","?")
        tg     = row.get("thoracic_grade",    "?")
        lumb_s = f"{lumb:.2f}°" if isinstance(lumb, float) else str(lumb)
        thor_s = f"{thor:.2f}°" if isinstance(thor, float) else str(thor)

        ax.set_title(
            f"{PANEL_TITLES[cat]}\n"
            f"Lumbar: {lumb_s} ({lp})  [{lg}]\n"
            f"Thoracic: {thor_s} ({tp})  [{tg}]",
            color="white", fontsize=11, fontweight="bold", pad=10)

        ax.text(0.03, 0.015, PANEL_TITLES[cat], transform=ax.transAxes,
                color="white", fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.38",
                          facecolor=PANEL_COLORS[cat],
                          edgecolor="white", alpha=0.92),
                va="bottom")
        add_legend(ax, lbl_sl)

    fig.suptitle(
        "Automated Cobb Angle Measurement — VerseFusion Dataset  (v3)\n"
        "CT + segmentation overlay  |  Red/Blue = Lumbar  |  Orange/Cyan = Thoracic\n"
        f"Iterative plane-normal endplate isolation  |  "
        f"Thoracic pairs ≥{MIN_THORACIC_SPAN}-vertebra span",
        color="white", fontsize=12, fontweight="bold", y=1.01)

    out_path = out_dir / "fig_comparison_verse_v3.png"
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER
# ═══════════════════════════════════════════════════════════════════════════════

def process_one_case(args_tuple):
    case_dir, out_dir_str, no_viz = args_tuple
    case_dir = Path(case_dir)
    out_dir  = Path(out_dir_str)
    case_id  = case_dir.name

    ct_path  = case_dir / "ct.nii.gz"
    msk_path = case_dir / "mask.nii.gz"
    if not ct_path.exists() or not msk_path.exists():
        return None, case_id, "Missing ct.nii.gz or mask.nii.gz"

    try:
        import nibabel as nib
        ct_img  = nib.load(str(ct_path))
        msk_img = nib.load(str(msk_path))
        ct_vol  = np.asarray(ct_img.get_fdata(),  dtype=np.float32)
        lbl_vol = np.asarray(msk_img.get_fdata(), dtype=np.uint8)
        affine  = msk_img.affine.astype(np.float64)
    except Exception as e:
        return None, case_id, f"Load error: {e}"

    if ct_vol.shape != lbl_vol.shape:
        return None, case_id, \
               f"Shape mismatch: CT {ct_vol.shape} vs mask {lbl_vol.shape}"

    try:
        row, pca = analyse_case(lbl_vol, affine, case_id)
    except Exception as e:
        return None, case_id, f"Analysis error: {e}"

    if not no_viz and pca:
        vp = out_dir / f"cobb_viz_{case_id}.png"
        try:
            visualise_case(lbl_vol, ct_vol, affine, row, pca, vp)
        except Exception:
            pass

    return row, case_id, None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

CSV_COLS = [
    "case_id", "n_vertebrae", "labels_found",
    "lumbar_cobb_deg", "lumbar_cobb_pair", "lumbar_grade",
    "lumbar_L1_S1_ref", "lumbar_L1_L5_ref",
    "thoracic_cobb_deg", "thoracic_cobb_pair", "thoracic_grade",
    "thoracic_T4_T12_ref",
    "combined_cobb_deg", "combined_cobb_pair", "combined_grade",
]


def main():
    global MIN_THORACIC_SPAN
    p = argparse.ArgumentParser(
        description="VerseFusion Cobb v3 — iterative plane-normal endplate isolation")
    p.add_argument("--compare",          action="store_true")
    p.add_argument("--csv",              default=None)
    p.add_argument("--data",             default=None, type=Path)
    p.add_argument("--out",              default="versefusion_figures_v3", type=Path)
    p.add_argument("--normal-case",      default=DEFAULT_CASES["normal"])
    p.add_argument("--lumbar-case",      default=DEFAULT_CASES["lumbar"])
    p.add_argument("--thoracic-case",    default=DEFAULT_CASES["thoracic"])
    p.add_argument("--data-dir",         default=None, type=Path)
    p.add_argument("--output-dir",       default="versefusion_results_v3", type=Path)
    p.add_argument("--workers",          type=int, default=None)
    p.add_argument("--no-viz",           action="store_true")
    p.add_argument("--no-progress",      action="store_true")
    p.add_argument("--limit",            type=int, default=None)
    p.add_argument("--min-thoracic-span",type=int, default=MIN_THORACIC_SPAN)
    p.add_argument("--ep-frac",          type=float, default=0.15,
                   help="Fraction of voxels used as endplate surface (default 0.15)")
    p.add_argument("--ep-iters",         type=int, default=3,
                   help="Plane-normal refinement iterations (default 3)")
    args = p.parse_args()

    MIN_THORACIC_SPAN = args.min_thoracic_span

    # Patch isolate_endplate_v3 defaults from CLI if provided
    import functools
    orig_iso = isolate_endplate_v3
    globals()["isolate_endplate_v3"] = functools.partial(
        orig_iso, frac=args.ep_frac, n_iter=args.ep_iters)

    # ── Comparison mode ───────────────────────────────────────────────────────
    if args.compare:
        if not args.csv or not args.data:
            p.error("--compare requires --csv and --data")
        args.out.mkdir(parents=True, exist_ok=True)
        scans_dir = Path(args.data)
        if scans_dir.name != "scans":
            scans_dir = scans_dir / "scans"
        make_comparison_figure(
            csv_path      = Path(args.csv),
            scans_dir     = scans_dir,
            out_dir       = args.out,
            case_normal   = args.normal_case,
            case_lumbar   = args.lumbar_case,
            case_thoracic = args.thoracic_case,
        )
        return

    # ── Full analysis mode ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "cobb_analysis_v3.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )
    log = logging.getLogger("versefusion_v3")
    log.info(f"VerseFusion Cobb v3  |  "
             f"min_thoracic_span={MIN_THORACIC_SPAN}  "
             f"ep_frac={args.ep_frac}  ep_iters={args.ep_iters}")
    log.info(f"Output: {args.output_dir.resolve()}")

    scans_dir = args.data_dir / "scans"
    case_dirs = sorted([d for d in scans_dir.iterdir() if d.is_dir()])
    if args.limit:
        case_dirs = case_dirs[:args.limit]
    log.info(f"Found {len(case_dirs)} cases")

    n_workers = args.workers or min(48, os.cpu_count() or 4)
    csv_path  = args.output_dir / "cobb_versefusion_v3.csv"
    csv_lock  = threading.Lock()
    n_ok = n_skip = 0
    all_rows  = []

    work_items = [(str(d), str(args.output_dir), args.no_viz) for d in case_dirs]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        writer.writeheader(); fh.flush()

        pbar = tqdm(total=len(case_dirs), desc="Analysing",
                    unit="case", disable=args.no_progress, dynamic_ncols=True)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(process_one_case, item): item
                       for item in work_items}
            for fut in as_completed(futures):
                try:
                    row, case_id, err = fut.result()
                except Exception as e:
                    log.error(f"Worker crash: {e}"); n_skip += 1
                    pbar.update(1); continue

                if row is None:
                    log.warning(f"Skip {case_id}: {err}"); n_skip += 1
                else:
                    with csv_lock:
                        writer.writerow(row); fh.flush()
                    all_rows.append(row); n_ok += 1
                    log.info(
                        f"  {case_id} | "
                        f"L={row.get('lumbar_cobb_deg','N/A')}°"
                        f" ({row.get('lumbar_grade','?')}) | "
                        f"T={row.get('thoracic_cobb_deg','N/A')}°"
                        f" ({row.get('thoracic_grade','?')})")
                pbar.update(1)
                pbar.set_postfix(ok=n_ok, skip=n_skip)
        pbar.close()

    log.info(f"\nDone: {n_ok} processed, {n_skip} skipped.")
    log.info(f"CSV: {csv_path.resolve()}")

    if all_rows:
        for label, key, gkey in [
                ("LUMBAR",   "lumbar_cobb_deg",  "lumbar_grade"),
                ("THORACIC", "thoracic_cobb_deg", "thoracic_grade")]:
            angles = [r[key] for r in all_rows if r.get(key) is not None]
            if not angles: continue
            a      = np.array(angles, dtype=float)
            grades = {}
            for r in all_rows:
                g = r.get(gkey)
                if g: grades[g] = grades.get(g, 0) + 1
            log.info(
                f"\n{label} (n={len(a)}): "
                f"mean={a.mean():.2f}°±{a.std():.2f}°  "
                f"median={np.median(a):.2f}°  "
                f"range={a.min():.2f}°–{a.max():.2f}°")
            log.info(f"  Grades: {grades}")


if __name__ == "__main__":
    main()
