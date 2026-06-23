"""ostk.geometry — pure, deterministic geometric primitives (world-mm space).

Every function is stateless and operates on plain numpy arrays, so they are
reproducible (no RNG, fixed sign conventions for the eigenvector/normal
sign ambiguity), low-latency (vectorised / closed-form fits, no per-voxel
Python loops), and picklable for process-pool parallelism.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# World "superior" direction for NIfTI affines mapped to RAS+ (nibabel default):
# +Z is cranial. Pass a data-derived axis instead where you don't trust this.
WORLD_SUPERIOR = np.array([0.0, 0.0, 1.0])


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _orient(vec: np.ndarray) -> np.ndarray:
    """Fix eigenvector sign ambiguity deterministically: make the
    largest-magnitude component positive (so repeated runs are identical)."""
    k = int(np.argmax(np.abs(vec)))
    return vec * (1.0 if vec[k] >= 0 else -1.0)


def principal_axes(points) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA of an (N,3) world point cloud.

    Returns (axes, eigenvalues, mean): `axes` is 3x3 with columns ordered by
    DESCENDING variance (axes[:,0] = long axis), each sign-fixed; eigenvalues
    descending; mean is the centroid.
    """
    P = np.asarray(points, dtype=np.float64)
    m = P.mean(axis=0)
    C = np.cov((P - m).T)
    w, V = np.linalg.eigh(C)                      # ascending eigenvalues
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    V = np.column_stack([_orient(V[:, i]) for i in range(V.shape[1])])
    return V, w, m


def fit_plane_tls(points) -> Tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares plane through an (N,3) cloud.

    Returns (point_on_plane=centroid, unit normal (sign-fixed), rms_residual).
    The normal is the eigenvector of the smallest covariance eigenvalue.
    """
    P = np.asarray(points, dtype=np.float64)
    m = P.mean(axis=0)
    C = np.cov((P - m).T)
    _, V = np.linalg.eigh(C)
    n = unit(_orient(V[:, 0]))
    rms = float(np.sqrt(np.mean(((P - m) @ n) ** 2)))
    return m, n, rms


def fit_sphere(points) -> Tuple[np.ndarray, float, float]:
    """Algebraic least-squares sphere (Kåsa/Coope) — closed form, robust to a
    PARTIAL sphere (FOV-clipped femoral head). Returns (center, radius, rms).

    Solves |p|^2 = 2 c·p + (r^2 - |c|^2) for c and the constant via lstsq.
    """
    P = np.asarray(points, dtype=np.float64)
    A = np.c_[2.0 * P, np.ones(len(P))]
    b = np.sum(P ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
    return c, r, rms


def angle_between(v1, v2, degrees: bool = True) -> float:
    """Unsigned angle between two vectors (directed)."""
    a, b = unit(v1), unit(v2)
    cos = float(np.clip(a @ b, -1.0, 1.0))
    ang = float(np.arccos(cos))
    return np.degrees(ang) if degrees else ang


def project_out(vectors, axis) -> np.ndarray:
    """Remove the component along `axis` (project onto the plane ⟂ axis).
    Accepts a single (3,) vector or an (N,3) array."""
    a = unit(axis)
    V = np.asarray(vectors, dtype=np.float64)
    if V.ndim == 2:
        return V - np.outer(V @ a, a)
    return V - (V @ a) * a


def signed_angle_in_plane(v1, v2, plane_normal, degrees: bool = True) -> float:
    """Signed angle from v1 to v2 measured *within* the plane ⟂ `plane_normal`
    (both vectors are projected into that plane first). Sign follows the
    right-hand rule about `plane_normal`. Use for per-segment lordosis where the
    direction of tilt (lordotic vs kyphotic) matters."""
    n = unit(plane_normal)
    a = unit(project_out(v1, n))
    b = unit(project_out(v2, n))
    s = float(np.cross(a, b) @ n)
    c = float(np.clip(a @ b, -1.0, 1.0))
    ang = float(np.arctan2(s, c))
    return np.degrees(ang) if degrees else ang


def cobb_angle(normal_a, normal_b, view_normal) -> float:
    """Cobb angle (deg) between two endplate PLANES as seen in the viewing plane
    (normal `view_normal`): project both endplate normals into that plane and take
    the angle between the planes — identical to drawing perpendiculars to each
    endplate and measuring their intersection. Sagittal view (`view_normal` = L–R
    axis) gives lordosis/kyphosis; coronal view (A–P axis) gives scoliosis.

    Orientation-INDEPENDENT: it uses |cos| so the result is the acute dihedral
    (0–90°) regardless of how each normal is oriented. This matters because
    `ostk.spine.fit_endplate` returns OUTWARD normals (superior→cranial,
    inferior→caudal); comparing a superior/inferior pair (e.g. vertebral wedging)
    with the raw directed angle would give the supplement. Both endplates of a
    realistic spinal angle are <90° apart, so the acute value is the true tilt."""
    a = unit(project_out(normal_a, view_normal))
    b = unit(project_out(normal_b, view_normal))
    cos = abs(float(np.clip(a @ b, -1.0, 1.0)))
    return float(np.degrees(np.arccos(cos)))


def rotation_matrix(axis, theta_rad: float) -> np.ndarray:
    """3×3 rotation by `theta_rad` about `axis` (right-hand rule), via Rodrigues.
    Used to rotate a mobile spinal segment about the patient L–R axis when
    synthesising a post-osteotomy correction (see ostk.surgery)."""
    a = unit(axis)
    x, y, z = a
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    C = 1.0 - c
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
