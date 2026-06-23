import numpy as np

from ostk import geometry as g
from ostk import metrics


# --- geometry: Cobb / signed angle -----------------------------------------

def _endplate_normal(tilt_deg):
    """Cranially-oriented endplate normal tilted by `tilt_deg` about the L–R (X)
    axis, lying in the patient sagittal (Y–Z) plane. +tilt leans toward +Y."""
    a = np.deg2rad(tilt_deg)
    return np.array([0.0, np.sin(a), np.cos(a)])


def test_cobb_angle_is_tilt_difference_in_sagittal_plane():
    lr = np.array([1.0, 0.0, 0.0])
    n_top, n_bot = _endplate_normal(15.0), _endplate_normal(-25.0)
    assert abs(g.cobb_angle(n_top, n_bot, lr) - 40.0) < 1e-6


def test_cobb_angle_ignores_out_of_plane_component():
    # an L–R (out-of-sagittal-plane) wobble must not change the sagittal Cobb
    lr = np.array([1.0, 0.0, 0.0])
    base = _endplate_normal(10.0)
    wobbled = g.unit(base + np.array([0.3, 0.0, 0.0]))
    assert abs(g.cobb_angle(base, wobbled, lr)) < 1e-6


def test_signed_angle_sign_follows_right_hand_rule():
    lr = np.array([1.0, 0.0, 0.0])
    a, b = _endplate_normal(15.0), _endplate_normal(-25.0)
    # cross(a,b)·X = sin(tilt_a - tilt_b) -> signed angle = tilt_a - tilt_b
    assert abs(g.signed_angle_in_plane(a, b, lr) - 40.0) < 1e-6
    assert abs(g.signed_angle_in_plane(b, a, lr) + 40.0) < 1e-6


# --- lumbar lordosis core --------------------------------------------------

def test_lumbar_lordosis_total_and_segments():
    lr = np.array([1.0, 0.0, 0.0])
    tilts = np.linspace(15.0, -25.0, len(metrics.LL_ENDPLATE_CHAIN))   # 6 levels
    normals = {lv: _endplate_normal(t)
               for lv, t in zip(metrics.LL_ENDPLATE_CHAIN, tilts)}
    r = metrics.lumbar_lordosis(normals, lr)
    assert abs(r["LL"] - 40.0) < 1e-6                 # L1 sup -> S1 sup Cobb
    assert r["span"] == "L1-S1"
    seg = r["segments"]
    assert len(seg) == len(metrics.LL_ENDPLATE_CHAIN) - 1
    assert abs(sum(seg.values()) - 40.0) < 1e-6       # signed segments sum to total
    for v in seg.values():                            # uniform tilt -> equal steps
        assert abs(v - 8.0) < 1e-6


def test_lumbar_lordosis_needs_two_levels():
    lr = np.array([1.0, 0.0, 0.0])
    r = metrics.lumbar_lordosis({"L1": _endplate_normal(10.0)}, lr)
    assert r["LL"] is None


# --- alignment targets / SRS-Schwab modifiers (Greenberg §73.6–73.7) -------

def test_pi_ll_mismatch_targets_and_grades():
    ok = metrics.pi_ll_mismatch(50.0, 45.0)           # mismatch 5°
    assert ok["pi_minus_ll"] == 5.0
    assert ok["within_target_9deg"] and not ok["surgical_target"]
    assert ok["schwab_modifier"] == "0" and ok["ll_shortfall_deg"] == 0.0

    mod = metrics.pi_ll_mismatch(60.0, 40.0)          # mismatch 20°
    assert mod["surgical_target"] and mod["schwab_modifier"] == "+"
    assert mod["ll_shortfall_deg"] == 11.0            # 20 - 9

    sev = metrics.pi_ll_mismatch(70.0, 40.0)          # mismatch 30°
    assert sev["schwab_modifier"] == "++"


def test_ll_increase_needed_eq_73_1():
    # ΔLL = (PI-LL-9) + (PT-20), clamped per term
    assert metrics.ll_increase_needed(60.0, 40.0, 25.0) == 16.0   # 11 + 5
    assert metrics.ll_increase_needed(55.0, 50.0, 15.0) == 0.0    # both terms ≤0
    assert metrics.ll_increase_needed(60.0, 40.0, 18.0) == 11.0   # PT term clamped


def test_schwab_sagittal_modifiers():
    r = metrics.schwab_sagittal_modifiers(60.0, 40.0, 25.0)       # no SVA
    assert r["PI-LL"] == "+" and r["PT"] == "+"
    assert r["SVA"] == "out_of_scope"
    assert r["objectives"]["LL=PI±9°"] is False
    assert r["objectives"]["PT<20°"] is False
    assert r["objectives"]["SVA<5cm"] is None
    assert r["ll_increase_needed_deg"] == 16.0

    balanced = metrics.schwab_sagittal_modifiers(50.0, 45.0, 10.0, sva_cm=3.0)
    assert balanced["PI-LL"] == "0" and balanced["PT"] == "0"
    assert balanced["SVA"] == "0"
    assert balanced["objectives"]["SVA<5cm"] is True


def test_pt_modifier_boundaries():
    assert metrics.schwab_sagittal_modifiers(50, 48, 19)["PT"] == "0"
    assert metrics.schwab_sagittal_modifiers(50, 48, 30)["PT"] == "+"
    assert metrics.schwab_sagittal_modifiers(50, 48, 31)["PT"] == "++"


def test_pi_ll_target_range():
    assert metrics.pi_ll_mismatch(50.0, 46.0)["ll_target_deg"] == [41.0, 59.0]


def test_pi_magnitude_category():
    assert metrics.pi_magnitude_category(40) == "low"
    assert metrics.pi_magnitude_category(45) == "average"
    assert metrics.pi_magnitude_category(60) == "average"
    assert metrics.pi_magnitude_category(61) == "high"


def test_roussouly_type_from_ss():
    assert metrics.roussouly_type_from_ss(32) == "1-2"
    assert metrics.roussouly_type_from_ss(35) == "3"
    assert metrics.roussouly_type_from_ss(45) == "3"
    assert metrics.roussouly_type_from_ss(46) == "4"


def test_surgical_recommendation_from_chapter():
    # near-balanced (case 0003): ΔLL=(55.5-45.7-9)+(23.2-20)=4.0° -> interbody, no osteotomy
    r = metrics.surgical_recommendation(55.5, 45.7, 23.2)
    assert round(r["ll_to_restore_deg"], 1) == 4.0
    assert r["severity"] == "mild" and r["osteotomy"] is None
    assert "interbody" in r["primary"].lower()

    # fully balanced: ΔLL=0 -> no realignment, standalone feasible (PT<20)
    bal = metrics.surgical_recommendation(50, 48, 12)
    assert bal["ll_to_restore_deg"] == 0.0 and "no major realignment" in bal["primary"]
    assert "standalone" in bal["fixation"]

    # moderate need ΔLL=11 -> ACR (≤12°/level)
    mod = metrics.surgical_recommendation(60, 42, 22)
    assert mod["osteotomy"] == "ACR (anterior, ALL release)"

    # severe ΔLL=46, |PI-LL|=40 -> PSO + open pelvic fixation
    sev = metrics.surgical_recommendation(70, 30, 35)
    assert sev["severity"] == "severe" and sev["osteotomy"] == "PSO"
    assert "ilium" in sev["fixation"]


def test_femoral_head_center_rejects_neck_and_shaft():
    """The robust head fit must recover the spherical head centre despite the neck
    and shaft, using the acetabular interface — and beat a naive whole-femur fit."""
    from ostk.labels import lid
    D = 96
    ijk = np.argwhere(np.ones((D, D, D), dtype=bool)).astype(float)
    label = np.zeros((D, D, D), dtype=np.int32); flat = label.reshape(-1)
    Chead = np.array([40.0, 48.0, 60.0]); R = 12.0
    # femur = head ball + neck/shaft cylinder running infero-laterally
    head = np.linalg.norm(ijk - Chead, axis=1) <= R
    axisv = g.unit(np.array([1.0, 0.0, -1.5]))
    d = (ijk - Chead) @ axisv
    inplane = np.linalg.norm((ijk - Chead) - np.outer(d, axisv), axis=1)
    shaft = (d >= 0) & (d <= 40) & (inplane <= 6.0)
    flat[head | shaft] = lid("femur_left")
    # acetabulum = thin shell hugging the superior head surface (the socket)
    rr = np.linalg.norm(ijk - Chead, axis=1)
    flat[(rr > R + 0.5) & (rr <= R + 3.0) & (ijk[:, 2] > Chead[2])] = lid("left_hip")

    c, r, rms = metrics.femoral_head_center(label, np.eye(4), "femur_left", "left_hip")
    assert np.linalg.norm(c - Chead) < 3.0          # recovered the head centre
    assert abs(r - R) < 3.0 and rms < 4.0           # anatomic radius, tight shell fit
    cn, _, _ = g.fit_sphere(ijk[head | shaft])      # naive fit is dragged toward shaft
    assert np.linalg.norm(c - Chead) < np.linalg.norm(cn - Chead)


# --- end-to-end from a synthetic label volume ------------------------------

def _ball(grid_pts, center, radius):
    return np.linalg.norm(grid_pts - center, axis=1) <= radius


def _body(grid_pts, center, normal, radius=18.0, half_height=7.0):
    """Solid short cylinder (a vertebral body) with its cranio-caudal axis along
    `normal` — so the cranial slab is a genuine 2-D endplate patch."""
    n = g.unit(normal)
    d = (grid_pts - center) @ n
    inplane = np.linalg.norm((grid_pts - center) - np.outer(d, n), axis=1)
    return (np.abs(d) <= half_height) & (inplane <= radius)


def test_lumbar_lordosis_from_label_phantom_plumbing():
    """Integration smoke test of the full `_from_label` chain (femoral-head
    sagittal axis -> per-level endplate fits -> LL). `_from_label` is the
    approximate, manually-validated glue layer (SPEC §5/§7) — so this asserts the
    plumbing (levels found, femurs used, a sane positive LL + 5 segments), not
    sub-degree accuracy, which is what the geometry-core tests above pin down."""
    from ostk.labels import lid

    D = 96
    ijk = np.argwhere(np.ones((D, D, D), dtype=bool)).astype(float)
    label = np.zeros((D, D, D), dtype=np.int32)
    flat = label.reshape(-1)

    # femoral heads -> L–R (sagittal) axis ~ +X (identity affine: world == index)
    flat[_ball(ijk, np.array([22.0, 48.0, 22.0]), 13.0)] = lid("femur_left")
    flat[_ball(ijk, np.array([74.0, 48.0, 22.0]), 13.0)] = lid("femur_right")

    # six lordotic bodies L1..S1 stacked in Z, tilting in the sagittal (Y–Z)
    # plane from +12° (L1) to -18° (S1).
    tilts = np.linspace(12.0, -18.0, len(metrics.LL_ENDPLATE_CHAIN))
    zs = np.linspace(80.0, 40.0, len(metrics.LL_ENDPLATE_CHAIN))
    for lv, t, z in zip(metrics.LL_ENDPLATE_CHAIN, tilts, zs):
        a = np.deg2rad(t)
        normal = np.array([0.0, np.sin(a), np.cos(a)])
        flat[_body(ijk, np.array([48.0, 48.0, z]), normal)] = lid(lv)

    m = metrics.lumbar_lordosis_from_label(np.asarray(label), np.eye(4),
                                           case_id="phantom")
    assert m.parameter == "lumbar_lordosis" and m.supine_ct is True
    assert "sagittal_ref_fallback" not in m.qc_flags          # femurs were found
    assert not any(f.startswith("missing_label") for f in m.qc_flags)
    assert m.value is not None and 5.0 < m.value < 80.0        # plausible, lordotic
    segs = m.landmarks_world_mm["per_segment_lordosis_deg"]
    assert len(segs) == len(metrics.LL_ENDPLATE_CHAIN) - 1


def _phantom_spine(D=96):
    """Femurs (-> L-R axis) + lordotic L1..S1 bodies on an identity grid."""
    from ostk.labels import lid
    ijk = np.argwhere(np.ones((D, D, D), dtype=bool)).astype(float)
    label = np.zeros((D, D, D), dtype=np.int32)
    flat = label.reshape(-1)
    flat[_ball(ijk, np.array([22.0, 48.0, 22.0]), 13.0)] = lid("femur_left")
    flat[_ball(ijk, np.array([74.0, 48.0, 22.0]), 13.0)] = lid("femur_right")
    tilts = np.linspace(12.0, -18.0, len(metrics.LL_ENDPLATE_CHAIN))
    zs = np.linspace(80.0, 40.0, len(metrics.LL_ENDPLATE_CHAIN))
    for lv, t, z in zip(metrics.LL_ENDPLATE_CHAIN, tilts, zs):
        a = np.deg2rad(t)
        normal = np.array([0.0, np.sin(a), np.cos(a)])
        flat[_body(ijk, np.array([48.0, 48.0, z]), normal)] = lid(lv)
    return label


def test_simulate_correction_adds_lordosis_pelvis_fixed():
    """Phase-1 post-op synthesis: rotating the segment at/above L3 by Δ° must raise
    the re-measured LL by ~Δ while the pelvis (PI/SS/PT) is untouched — so PI−LL
    (the surgical target) improves by ~Δ."""
    from ostk import surgery
    label, A = _phantom_spine(), np.eye(4)
    pre = metrics.spinopelvic_summary_from_label(label, A, case_id="pre")
    assert pre["LL"] is not None and pre["PI"] is not None

    DELTA = 12.0
    out = surgery.simulate_correction(label, A, "L3", DELTA)
    post = metrics.spinopelvic_summary_from_label(out, A, case_id="post")

    assert abs((post["LL"] - pre["LL"]) - DELTA) < 3.0          # LL += Δ
    assert abs(post["PI"] - pre["PI"]) < 1.0                    # pelvis fixed
    assert abs(post["SS"] - pre["SS"]) < 1.0
    assert abs(post["PT"] - pre["PT"]) < 1.0
    improved = pre["PI-LL"]["pi_minus_ll"] - post["PI-LL"]["pi_minus_ll"]
    assert improved > DELTA - 3.0                               # PI−LL closes by ~Δ
