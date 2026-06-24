"""tests/test_cobb.py — synthetic ground-truth test for ostk.cobb.

No real CT data needed: builds a synthetic label volume with a known coronal
tilt (S1 tilted -5deg, L1 tilted +15deg, L2-L5 untilted) and checks the
computed Cobb angle against that ground truth (|15 - (-5)| = 20deg).
"""
import numpy as np
import pytest

from ostk.cobb import coronal_cobb_from_label

SHAPE = (300, 150, 260)  # RL, AP, SI (identity affine -> voxel idx == world mm)


def _rotate_y(theta_deg):
    th = np.radians(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _stamp_box(label_vol, center, half_extent=(15, 10, 7), theta_deg=0.0,
              label_id=1, margin=8):
    """Rasterize a SOLID, gap-free rotated box (inverse-rotate each candidate
    voxel back to the box's local frame and test membership) so the result is
    6-connected -- a sparse rounded point cloud breaks largest_component."""
    R_inv = _rotate_y(-theta_deg)
    reach = max(half_extent) + margin
    cx, cy, cz = (int(round(v)) for v in center)
    x0, x1 = max(cx - reach, 0), min(cx + reach, SHAPE[0])
    y0, y1 = max(cy - reach, 0), min(cy + reach, SHAPE[1])
    z0, z1 = max(cz - reach, 0), min(cz + reach, SHAPE[2])
    xs, ys, zs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1),
                             np.arange(z0, z1), indexing="ij")
    voxel_idx = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    local = (voxel_idx - np.array(center)) @ R_inv.T
    inside = (np.abs(local[:, 0]) <= half_extent[0]) & \
             (np.abs(local[:, 1]) <= half_extent[1]) & \
             (np.abs(local[:, 2]) <= half_extent[2])
    hit = voxel_idx[inside]
    label_vol[hit[:, 0], hit[:, 1], hit[:, 2]] = label_id


def _stamp_sphere(label_vol, center, radius=15.0, label_id=11, margin=2):
    reach = int(round(radius)) + margin
    cx, cy, cz = (int(round(v)) for v in center)
    x0, x1 = max(cx - reach, 0), min(cx + reach, SHAPE[0])
    y0, y1 = max(cy - reach, 0), min(cy + reach, SHAPE[1])
    z0, z1 = max(cz - reach, 0), min(cz + reach, SHAPE[2])
    xs, ys, zs = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1),
                             np.arange(z0, z1), indexing="ij")
    voxel_idx = np.stack([xs, ys, zs], axis=-1).reshape(-1, 3)
    d2 = np.sum((voxel_idx - np.array(center)) ** 2, axis=1)
    hit = voxel_idx[d2 <= radius ** 2]
    label_vol[hit[:, 0], hit[:, 1], hit[:, 2]] = label_id


@pytest.fixture
def synthetic_label_vol():
    """S1 tilted -5deg, L1 tilted +15deg, L2-L5 untilted, plus two femoral-head
    spheres for the L-R axis derivation. Ground-truth Cobb (L1 vs S1, the
    default full lumbosacral span) = |15 - (-5)| = 20deg."""
    label_vol = np.zeros(SHAPE, dtype=np.int32)
    affine = np.eye(4)
    levels = [(7, 50, -5.0), (5, 80, 0.0), (4, 110, 0.0),
             (3, 140, 0.0), (2, 170, 0.0), (1, 200, 15.0)]
    for label_id, z, theta in levels:
        _stamp_box(label_vol, center=(150, 75, z), theta_deg=theta, label_id=label_id)
    _stamp_sphere(label_vol, (110, 75, 20), radius=15, label_id=11)  # femur_left
    _stamp_sphere(label_vol, (190, 75, 20), radius=15, label_id=12)  # femur_right
    return label_vol, affine


def test_cobb_matches_ground_truth(synthetic_label_vol):
    label_vol, affine = synthetic_label_vol
    m = coronal_cobb_from_label(label_vol, affine, case_id="synthetic")
    assert m.value is not None
    assert abs(m.value - 20.0) < 2.0, f"expected ~20deg, got {m.value}"
    assert m.landmarks_world_mm["cobb_top_level"] == "L1"
    assert m.landmarks_world_mm["cobb_bottom_level"] == "S1"
    assert m.qc_flags == ["ok"]


def test_cobb_falls_back_gracefully_without_femurs(synthetic_label_vol):
    """No femurs -> can't derive a patient L-R axis -> falls back to image-X
    with a flag, but should still return a value rather than crashing."""
    label_vol, affine = synthetic_label_vol
    label_vol = label_vol.copy()
    label_vol[label_vol == 11] = 0
    label_vol[label_vol == 12] = 0
    m = coronal_cobb_from_label(label_vol, affine, case_id="synthetic_no_femur")
    assert m.value is not None
    assert "sagittal_ref_fallback" in m.qc_flags


def test_cobb_no_levels_returns_none_not_crash():
    """An empty label volume should degrade gracefully (None + flag), per
    SPEC §4: never silently drop a bad case."""
    label_vol = np.zeros(SHAPE, dtype=np.int32)
    affine = np.eye(4)
    m = coronal_cobb_from_label(label_vol, affine, case_id="empty")
    assert m.value is None
    assert "no_levels_available" in m.qc_flags
