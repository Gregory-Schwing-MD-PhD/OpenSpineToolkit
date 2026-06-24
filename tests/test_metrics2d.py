"""2-D spinopelvic port — proves the angles MATCH the 3-D pipeline on identical geometry
(no accuracy drop), the PI = SS + PT identity holds, the 2-D mask extractor works, and the
dimension-aware dispatcher routes/overrides correctly."""
import numpy as np
import pytest

from ostk import metrics, metrics2d
from ostk.geometry import cobb_angle


# sagittal frame for these tests: L–R axis = +X, so the sagittal plane is (Y, Z) with
# Y = anterior and Z = superior. A 3-D vector/point projects to 2-D as (Y, Z).
def _proj(p):
    return np.array([p[1], p[2]])


def _endplate_normal_3d(slope_deg):
    """S1/L1 endplate normal in the sagittal plane for a given endplate tilt (deg)."""
    a = np.radians(slope_deg)
    return np.array([0.0, -np.sin(a), np.cos(a)])


def test_2d_pi_ss_pt_match_3d_and_truth():
    """The real 3-D function (_pi_from_plane) and the 2-D port must return the SAME PI/SS/PT
    for the same endplate + femoral geometry — and both must equal the constructed truth."""
    SS_t, PT_t = 35.0, 20.0
    n = _endplate_normal_3d(SS_t)                       # S1 endplate normal
    m = np.array([0.0, 30.0, 100.0])                   # S1 endplate midpoint
    radius = 120.0 * np.array([0.0, np.sin(np.radians(PT_t)), np.cos(np.radians(PT_t))])
    bicox = m - radius                                  # hip axis so radius hits PT_t
    cL, cR = bicox - [40, 0, 0], bicox + [40, 0, 0]

    r3 = metrics._pi_from_plane(m, n, 0.0, cL, cR)      # ← actual 3-D code

    e2 = metrics2d._perp(_proj(n))                      # endplate direction in 2-D
    m2 = _proj(m)
    s2 = metrics2d.spinopelvic_summary_2d(
        {"S1": (m2 - 30 * e2, m2 + 30 * e2)}, _proj(bicox), sup=(0, 1))

    assert abs(r3["PI"] - (SS_t + PT_t)) < 1e-6        # 3-D sanity (PI = SS + PT)
    assert abs(s2["PI"] - r3["PI"]) < 0.05            # 2-D == 3-D  (no accuracy drop)
    assert abs(s2["SS"] - r3["SS"]) < 0.05
    assert abs(s2["PT"] - r3["PT"]) < 0.05
    assert abs(s2["PI"] - (SS_t + PT_t)) < 0.05       # 2-D == truth


def test_2d_ll_matches_3d_cobb():
    """LL (L1↔S1 Cobb) from the 2-D port must equal geometry.cobb_angle on the same normals."""
    tA, tB = 12.0, -18.0                                # L1, S1 endplate tilts
    nA, nB = _endplate_normal_3d(tA), _endplate_normal_3d(tB)
    LL_3d = cobb_angle(nA, nB, np.array([1.0, 0, 0]))

    eA, eB = metrics2d._perp(_proj(nA)), metrics2d._perp(_proj(nB))
    cA, cB = np.array([20.0, 200.0]), np.array([25.0, 100.0])
    out = metrics2d.lumbar_lordosis_2d(
        {"L1": (cA - 30 * eA, cA + 30 * eA), "S1": (cB - 30 * eB, cB + 30 * eB)})

    assert abs(out["LL"] - LL_3d) < 0.05
    assert abs(out["LL"] - abs(tA - tB)) < 0.05       # = 30°


@pytest.mark.parametrize("SS_t,PT_t", [(40, 10), (50, 25), (30, 5)])
def test_pi_identity_2d(SS_t, PT_t):
    n = _endplate_normal_3d(SS_t)
    m = np.array([0.0, 30.0, 100.0])
    bicox = m - 100.0 * np.array([0.0, np.sin(np.radians(PT_t)), np.cos(np.radians(PT_t))])
    e2 = metrics2d._perp(_proj(n))
    m2 = _proj(m)
    s = metrics2d.spinopelvic_summary_2d({"S1": (m2 - 30 * e2, m2 + 30 * e2)}, _proj(bicox))
    assert abs(s["PI"] - (s["SS"] + s["PT"])) < 0.05


def test_no_femoral_head_skips_pi_pt_keeps_ll_ss():
    """A lumbar film that crops the hips: PI/PT None, but SS and LL still compute."""
    s = metrics2d.spinopelvic_summary_2d(
        {"L1": ((0.0, 200.0), (40.0, 200.0)), "S1": ((0.0, 100.0), (40.0, 100.0))},
        femoral=None)
    assert s["PI"] is None and s["PT"] is None
    assert s["SS"] == 0.0 and s["LL"] == 0.0
    assert any("no_femoral_head" in f for f in s["qc_flags"])


def test_dispatcher_auto_and_override():
    line = ((0.0, 100.0), (40.0, 100.0))               # horizontal S1 (SS = 0)
    auto = metrics.spinopelvic_summary({"S1": line}, femoral=(20.0, 0.0))
    assert auto["modality"] == "radiograph_2d" and auto["SS"] == 0.0
    # forcing 3-D on 2-D data fails fast (override path exercised)
    with pytest.raises(ValueError):
        metrics.spinopelvic_summary({"S1": line}, mode="3d")
    # a 3-D label volume routes to the 3-D pipeline
    assert metrics._infer_mode(np.zeros((4, 4, 4))) == "3d"
    assert metrics._infer_mode(np.zeros((4, 4))) == "2d"


def test_2d_mask_extraction_plumbing():
    """A 2-D label mask routes through the dispatcher and yields finite, sane params."""
    from ostk.labels import LABELS
    m = np.zeros((220, 200), dtype=np.int32)
    m[150:170, 80:120] = LABELS["L1"]                  # superior body (high row = superior, sup=(0,1))
    m[60:80, 80:120] = LABELS["S1"]
    rr, cc = np.mgrid[0:220, 0:200]
    m[((rr - 30) ** 2 + (cc - 70) ** 2) < 18 ** 2] = LABELS["femur_left"]
    m[((rr - 30) ** 2 + (cc - 130) ** 2) < 18 ** 2] = LABELS["femur_right"]

    s = metrics.spinopelvic_summary(m, sup=(0, 1))     # ndim 2 -> 2-D
    assert s["modality"] == "radiograph_2d"
    assert s["LL"] is not None and abs(s["LL"]) < 5.0  # both endplates ~horizontal
    assert abs(s["SS"]) < 5.0
    assert s["PI"] is not None and np.isfinite(s["PI"])
