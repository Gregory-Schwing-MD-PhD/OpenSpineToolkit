"""
disc_spacing.py — Lumbar intervertebral disc height measurement
for the CTSpinoPelvic1K dataset.

Measures disc heights using 2D contour-based nearest-neighbor analysis
in the mid-sagittal slice, replicating how a radiologist places calipers:

  1. Isolate the vertebral BODY in the sagittal mask (erosion → largest CC →
     dilate back) to exclude posterior elements (pedicles, lamina, spinous
     processes, sacral arch) before any contour is extracted.
  2. Inferior contour of upper vertebral body : per-column max row.
  3. Superior contour of lower vertebral body : per-column min row.
  4. Restrict to the central 35 % of AP columns where a valid body-to-body
     gap exists — targets the mid-disc, avoids anterior osteophytes and
     residual posterior anatomy.
  5. From the median AP column of that window, find the nearest-neighbour
     point on the lower endplate — the line naturally tilts with the disc.
  6. Height = Euclidean mm distance between the two cortical surface points.

Labels: 1=L1, 2=L2, 3=L3, 4=L4, 5=L5, 7=Sacrum (S1)   (no L6)
Disc levels: L1-L2, L2-L3, L3-L4, L4-L5, L5-S1

Usage:
    python3 disc_spacing.py --root ~/Downloads/CTSpinoPelvic1K-Sample

Output:
    disc_spacing_results.csv   — one row per case per disc level (height only)
    disc_spacing_plot.png      — sagittal CT with colored masks and caliper lines
"""

import argparse
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ── Dataset import ────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
try:
    from dataset_interface import CTSpinoPelvic1K
except ImportError:
    print("ERROR: Could not find dataset_interface.py")
    sys.exit(1)

# ── Labels and disc levels ────────────────────────────────────────────────────
LABEL_NAMES = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 7: "Sacrum"}
DISC_LEVELS = [
    (1, 2, "L1-L2"),
    (2, 3, "L2-L3"),
    (3, 4, "L3-L4"),
    (4, 5, "L4-L5"),
    (5, 7, "L5-S1"),
]

# ── Sagittal slice extraction ─────────────────────────────────────────────────

def _axis_info(affine):
    """Return (si_vox_axis, lr_vox_axis, ap_vox_axis, si_sp, ap_sp, spacings)."""
    spacings    = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    si_vox_axis = int(np.argmax(np.abs(affine[2, :3])))
    lr_vox_axis = int(np.argmax(np.abs(affine[0, :3])))
    ap_vox_axis = [a for a in range(3) if a != si_vox_axis and a != lr_vox_axis][0]
    return (si_vox_axis, lr_vox_axis, ap_vox_axis,
            float(spacings[si_vox_axis]),
            float(spacings[ap_vox_axis]),
            spacings)


def _get_sagittal_display(label_array, affine):
    """
    Extract the mid-sagittal label slice arranged as [row=SI, col=AP]
    and crop to the spine bounding box + margin.

    Returns
    -------
    crop_label            : 2D ndarray  [row=SI, col=AP], origin='upper'
    r0, c0                : crop offsets into the full display slice
    si_sp, ap_sp          : mm per display row and column
    si_vox_axis, ap_vox_axis, lr_vox_axis
    """
    si_vox_axis, lr_vox_axis, ap_vox_axis, si_sp, ap_sp, _ = _axis_info(affine)

    spine_vox = np.argwhere((label_array >= 1) & (label_array <= 7))
    mid_lr    = int(np.median(spine_vox[:, lr_vox_axis]))

    sag_idx              = [slice(None), slice(None), slice(None)]
    sag_idx[lr_vox_axis] = mid_lr
    sag_label            = label_array[tuple(sag_idx)]

    remaining = [a for a in range(3) if a != lr_vox_axis]
    if remaining.index(si_vox_axis) != 0:
        sag_label = sag_label.T   # → [si_idx, ap_idx]

    si_vals = spine_vox[:, si_vox_axis]
    ap_vals = spine_vox[:, ap_vox_axis]
    si_pad  = int(0.08 * (si_vals.max() - si_vals.min()))
    ap_pad  = int(0.30 * (ap_vals.max() - ap_vals.min()))
    r0 = max(0,                    si_vals.min() - si_pad)
    r1 = min(sag_label.shape[0]-1, si_vals.max() + si_pad)
    c0 = max(0,                    ap_vals.min() - ap_pad)
    c1 = min(sag_label.shape[1]-1, ap_vals.max() + ap_pad)

    return (sag_label[r0:r1+1, c0:c1+1],
            int(r0), int(c0), si_sp, ap_sp,
            si_vox_axis, ap_vox_axis, lr_vox_axis)


# ── 2-D contour-based disc height measurement ─────────────────────────────────

def _isolate_vertebral_body(mask_2d, ap_sp, si_sp):
    """
    Isolate the vertebral body from a full vertebra sagittal mask.

    The full mask includes the body plus posterior elements (pedicles, lamina,
    spinous process, sacral arch).  We sever those thin connections via binary
    erosion, keep only the largest connected component (the body), then dilate
    back within the original mask boundary to restore the cortical rim.

    Progressive fallback: tries 5 mm → 3.5 mm → 2 mm erosion radii; if all
    over-erode (empty result) the original mask is returned unchanged.

    Parameters
    ----------
    mask_2d : 2-D bool array  [rows=SI, cols=AP]
    ap_sp   : mm per AP column (pixel width  in the display slice)
    si_sp   : mm per SI row   (pixel height in the display slice)
    """
    if not mask_2d.any():
        return mask_2d.copy()

    for erode_mm in (5.0, 3.5, 2.0):
        erode_ap = max(1, round(erode_mm / ap_sp))
        erode_si = max(1, round(erode_mm / si_sp))
        struct   = np.ones((erode_si, erode_ap), dtype=bool)
        eroded   = ndimage.binary_erosion(mask_2d, structure=struct)
        if not eroded.any():
            continue

        labeled, n_cc = ndimage.label(eroded)
        if n_cc == 0:
            continue

        sizes       = np.bincount(labeled.ravel())
        sizes[0]    = 0
        body_eroded = labeled == int(sizes.argmax())

        dilated   = ndimage.binary_dilation(body_eroded, structure=struct)
        candidate = dilated & mask_2d
        if candidate.any():
            return candidate.astype(bool)

    return mask_2d.copy()


def _measure_disc_2d(upper_mask, lower_mask, si_sp, ap_sp, central_frac=0.35):
    """
    Measure inter-endplate disc height from two 2-D boolean masks in the
    sagittal display slice (row=SI, col=AP, origin='upper').

    Algorithm
    ---------
    1. Isolate the vertebral body of each mask (removes pedicles, lamina,
       spinous processes, and sacral posterior arch) so that all subsequent
       contour extraction operates only on the endplate cortex.
    2. Inferior contour of upper vertebral body : max row per AP column.
    3. Superior contour of lower vertebral body : min row per AP column.
    4. Keep only AP columns where both body contours are present AND the
       disc gap is positive (open disc space between the two bodies).
    5. Restrict to the central `central_frac` of those valid AP columns,
       targeting the geometric mid-disc (avoids anterior osteophytes and
       any residual posterior anatomy at the body margin).
    6. At the median AP column of that window, take the upper endplate point
       and find its nearest neighbour on the lower endplate contour (in mm).
       The line naturally tilts with the local disc plane.
    7. Height = Euclidean mm distance between the two cortical surface points.

    Returns
    -------
    height_mm : float or None
    pt_upper  : (col, row) in cropped display coordinates, or None
    pt_lower  : (col, row) in cropped display coordinates, or None
    """
    # ── 1. Isolate vertebral bodies ──────────────────────────────────────────
    upper_body = _isolate_vertebral_body(upper_mask, ap_sp, si_sp)
    lower_body = _isolate_vertebral_body(lower_mask, ap_sp, si_sp)

    n_cols = upper_body.shape[1]

    # ── 2. Inferior contour of upper body (largest row = most inferior) ──────
    upper_inf = {}
    for col in range(n_cols):
        rows = np.where(upper_body[:, col])[0]
        if len(rows):
            upper_inf[col] = int(rows.max())

    # ── 3. Superior contour of lower body (smallest row = most superior) ─────
    lower_sup = {}
    for col in range(n_cols):
        rows = np.where(lower_body[:, col])[0]
        if len(rows):
            lower_sup[col] = int(rows.min())

    # ── 4. Valid columns: both body contours present, positive disc gap ───────
    valid_cols = sorted(
        c for c in set(upper_inf) & set(lower_sup)
        if lower_sup[c] > upper_inf[c]
    )
    if len(valid_cols) < 4:
        return None, None, None

    # ── 5. Central fraction of the valid AP body overlap ─────────────────────
    n  = len(valid_cols)
    lo = int(n * (0.5 - central_frac / 2))
    hi = int(n * (0.5 + central_frac / 2))
    hi = max(hi, lo + 2)
    central_cols = valid_cols[lo:hi] or valid_cols

    # ── 6. Contour arrays and mm conversion ──────────────────────────────────
    u_cont = np.array([(c, upper_inf[c]) for c in central_cols], dtype=float)
    l_cont = np.array([(c, lower_sup[c]) for c in central_cols], dtype=float)

    u_mm = u_cont * [ap_sp, si_sp]
    l_mm = l_cont * [ap_sp, si_sp]

    # ── 7. Nearest-neighbour from median upper endplate point ─────────────────
    mid_idx  = len(central_cols) // 2
    u_ref_mm = u_mm[mid_idx]

    dists  = np.sqrt(((l_mm - u_ref_mm) ** 2).sum(axis=1))
    j_near = int(np.argmin(dists))

    pt_upper = (u_cont[mid_idx, 0], u_cont[mid_idx, 1])   # (col, row)
    pt_lower = (l_cont[j_near,   0], l_cont[j_near,   1])
    height   = float(dists[j_near])

    return height, pt_upper, pt_lower


def measure_disc_heights_2d(label_array, affine):
    """
    Measure all disc levels using 2-D contour analysis in the mid-sagittal slice.

    Returns
    -------
    heights : dict  {disc_name: float_mm or None}
    viz_pts : dict  {disc_name: ((col_u, row_u), (col_l, row_l)) or None}
              All coordinates in the CROPPED display slice (col=AP, row=SI).
    """
    crop_label, _, _, si_sp, ap_sp, *_ = _get_sagittal_display(label_array, affine)

    heights = {}
    viz_pts = {}

    for upper_lbl, lower_lbl, disc_name in DISC_LEVELS:
        upper_mask = crop_label == upper_lbl
        lower_mask = crop_label == lower_lbl

        if not upper_mask.any() or not lower_mask.any():
            heights[disc_name] = None
            viz_pts[disc_name] = None
            continue

        h, pt_u, pt_l = _measure_disc_2d(upper_mask, lower_mask, si_sp, ap_sp)
        heights[disc_name] = h
        viz_pts[disc_name] = (pt_u, pt_l) if h is not None else None

    return heights, viz_pts



# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_case(case, disc_results, viz_pts, save_path="disc_spacing_plot.png"):
    """
    Sagittal CT slice with colored vertebra masks and contour-based
    disc height caliper lines.  Lines follow the local disc angle rather
    than being forced vertical.
    """
    from matplotlib.patches import Patch

    label_array, affine = case.load_label()
    ct_array = None
    try:
        ct_array, _ = case.load_ct()
    except Exception:
        pass

    present = [lbl for lbl in [1, 2, 3, 4, 5, 6, 7] if (label_array == lbl).any()]
    if not present:
        print(f"  No labels found for {case.token}")
        return

    (crop_label, r0, c0, si_sp, ap_sp,
     si_vox_axis, ap_vox_axis, lr_vox_axis) = _get_sagittal_display(label_array, affine)

    _, _, _, _, _, spacings = _axis_info(affine)
    asp = spacings[si_vox_axis] / spacings[ap_vox_axis]   # display aspect ratio

    # CT sagittal slice — same mid_lr and crop as label
    spine_vox = np.argwhere((label_array >= 1) & (label_array <= 7))
    mid_lr    = int(np.median(spine_vox[:, lr_vox_axis]))
    crop_ct   = None
    if ct_array is not None:
        sag_idx              = [slice(None), slice(None), slice(None)]
        sag_idx[lr_vox_axis] = mid_lr
        sag_ct               = ct_array[tuple(sag_idx)]
        remaining = [a for a in range(3) if a != lr_vox_axis]
        if remaining.index(si_vox_axis) != 0:
            sag_ct = sag_ct.T
        r1     = r0 + crop_label.shape[0] - 1
        c1     = c0 + crop_label.shape[1] - 1
        crop_ct = sag_ct[r0:r1+1, c0:c1+1]

    # Colored RGBA overlay
    colors_rgb = {
        1: (0.85, 0.75, 0.30),   # L1  gold
        2: (0.58, 0.26, 0.14),   # L2  brown
        3: (0.18, 0.58, 0.85),   # L3  blue
        4: (0.85, 0.22, 0.18),   # L4  red
        5: (0.18, 0.58, 0.90),   # L5  blue
        6: (0.55, 0.18, 0.78),   # L6  purple
        7: (0.28, 0.85, 0.28),   # Sacrum  green
    }
    rgba = np.zeros((*crop_label.shape, 4))
    for lbl, (r, g, b) in colors_rgb.items():
        rgba[crop_label == lbl] = (r, g, b, 0.65)

    fig, ax = plt.subplots(figsize=(6, 11))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    if crop_ct is not None:
        ax.imshow(crop_ct, cmap='gray', origin='upper',
                  vmin=-200, vmax=1000, aspect=asp)
    ax.imshow(rgba, origin='upper', aspect=asp)

    # ── Caliper annotations ────────────────────────────────────────────────
    for _, _, disc_name in DISC_LEVELS:
        val = disc_results.get(disc_name)
        pts = viz_pts.get(disc_name) if viz_pts else None
        if val is None or pts is None:
            continue

        pt_u, pt_l = pts                           # (col, row) in cropped display
        x1, y1 = float(pt_u[0]), float(pt_u[1])   # upper cortical surface
        x2, y2 = float(pt_l[0]), float(pt_l[1])   # lower cortical surface

        ax.plot([x1, x2], [y1, y2], '-', color='#ff6b9d', lw=2, zorder=5)
        ax.scatter([x1, x2], [y1, y2], color='#ff6b9d', s=55, zorder=6,
                   edgecolors='white', linewidths=0.5)
        x_text = max(x1, x2) + 6
        y_text = (y1 + y2) / 2
        ax.text(x_text, y_text, f'{disc_name}: {val:.1f} mm',
                color='#ff6b9d', fontsize=9, va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black',
                          alpha=0.55, edgecolor='none'))

    patches = [Patch(facecolor=colors_rgb.get(l, (0.5, 0.5, 0.5)),
                     label=LABEL_NAMES.get(l, str(l)))
               for l in present if l in colors_rgb]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#111', edgecolor='#555',
              labelcolor='white', fontsize=9)
    ax.set_title(f'Disc Heights — Token {case.token}',
                 color='white', fontsize=12, pad=8)
    ax.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  Saved plot: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Measure lumbar intervertebral disc heights.")
    parser.add_argument("--root",   required=True,
                        help="Path to your CTSpinoPelvic1K-Sample folder")
    parser.add_argument("--output", default="disc_spacing_results.csv")
    parser.add_argument("--plot",   default="disc_spacing_plot.png")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.root}")
    ds = CTSpinoPelvic1K(args.root)
    print(ds.stats())

    cases = ds.filter(config="fused", present_only=True)
    if not cases:
        cases = [c for c in ds.cases if c.exists()]
    if not cases:
        print("ERROR: No cases found. Check --root path.")
        sys.exit(1)

    print(f"\nProcessing {len(cases)} cases...\n")
    disc_level_names = [d[2] for d in DISC_LEVELS]

    all_rows = []
    plot_stem = Path(args.plot)

    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] Token: {case.token} ({case.config})")
        try:
            label_array, affine = case.load_label()
            disc_results, viz_pts = measure_disc_heights_2d(label_array, affine)

            row = {"token": case.token, "config": case.config}
            for name in disc_level_names:
                val      = disc_results.get(name)
                row[name] = f"{val:.2f}" if val is not None else "N/A"
                print(f"    {name}: {row[name]} mm")
            all_rows.append(row)

            if any(v is not None for v in disc_results.values()):
                plot_path = plot_stem.with_stem(f"{plot_stem.stem}_{case.token}")
                print(f"  Generating plot → {plot_path}")
                visualize_case(case, disc_results, viz_pts, save_path=str(plot_path))

        except Exception as e:
            print(f"    ERROR loading case {case.token}: {e}")
            continue

    if all_rows:
        fieldnames = ["token", "config"] + disc_level_names
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nNo results to save.")

    print("\nDone!")


if __name__ == "__main__":
    main()
