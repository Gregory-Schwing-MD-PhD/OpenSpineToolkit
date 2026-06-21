# OpenSpineToolbox — Programming Spec & Shared Mask Contract

This is the **authoritative technical contract** for miniprojects that compute
measurements from CTSpinoPelvic1K segmentation masks. The top-level
[README](README.md) covers Git/PR mechanics; *this* file covers **what the data
is, how to read it, what to output, and which parameters to build in what order.**

Read sections 1–5 before writing any code. Then find your parameter in section 6.

---

## 1. The data you're working with (CTSpinoPelvic1K v3)

Each case is a CT volume + a voxel-aligned integer label map:

```
ct/{NNNN}_ct.nii.gz        CT (Hounsfield units), PIR orientation
labels/{NNNN}_*.nii.gz     integer label map, same grid as the CT
```

**v3 label scheme (integer id → structure):**

| id | structure | id | structure |
|----|-----------|----|-----------|
| 0 | background | 11 | femur_left |
| 1–6 | L1–L6 | 12 | femur_right |
| 7 | S1 (carved from sacrum) | 13–25 | T1–T13 (FOV-limited) |
| 8 | sacrum (S2–S5 after S1 carve) | 26–49 | ribs (RESERVED, empty in v3) |
| 9 | left_hip | 50 | ignore (partial-annotation) |
| 10 | right_hip | | |

**Facts that govern every measurement:**

- **Supine CT.** Positional sagittal-balance values differ from standing
  radiographs. See the golden rule in §5.
- **FOV-limited cranially.** Thoracic GT reliably covers only ~T8→T12 (NOT T1),
  and there is **no C7/cervical**. Any parameter needing C7 or T1 is out of scope
  on most cases (§5, §6 Tier 3).
- **World coordinates come from the NIfTI affine.** Never measure in voxel
  index space — distances/angles must be in millimetres/degrees via the affine.
- **`ignore` (50)** marks partial/unlabeled regions — treat as "unknown," never
  as background.

---

## 2. Environment / dependencies (baseline)

Python ≥ 3.10. Recommended baseline (pin in your project's own `requirements.txt`):

```
nibabel>=5      # NIfTI read + affine
numpy>=1.24
scipy>=1.10     # least-squares fits, linear algebra, distance transforms
scikit-image>=0.22   # marching_cubes for sub-voxel surfaces, region props
SimpleITK>=2.3       # optional: resampling, connected components
```

Keep dependencies per-project (the repo intentionally has no shared env). If you
reuse a primitive from §3, copy it into your project or vendor a small helper —
don't add a repo-wide package without discussing it first.

---

## 3. Shared geometry contract (the primitives every project reuses)

Implement these once and reuse them; they are where fidelity is won or lost.

- **`mask(label, id) -> bool array`** — binary mask for a structure id.
- **`voxel_to_world(ijk, affine) -> xyz_mm`** — apply the affine; **all** geometry
  downstream is in world mm.
- **`centroid_world(mask, affine) -> xyz`** — mean of world coords of mask voxels.
- **`principal_axes(points) -> evecs, evals`** — PCA of a world point cloud (vertebral
  long axis, sacral axis, etc.).
- **`fit_plane_tls(points) -> point, normal`** — total-least-squares (PCA) plane
  through a surface point cloud (endplates). Report the RMS residual.
- **`fit_sphere(points) -> center, radius, residual`** — algebraic least-squares
  sphere fit (femoral heads). Robust to *partial* spheres, which matters when a
  head is FOV-clipped.
- **`endplate_surface(body_mask, axis, which) -> points`** — the superior or
  inferior articular surface of a vertebral body: take the body's cranio-caudal
  axis (its principal axis), then the thin voxel slab at the cranial (or caudal)
  extreme; for sub-voxel fidelity, run `marching_cubes` and keep faces whose
  normal aligns with the axis.
- **`patient_sagittal_plane(...)` (clever + robust):** derive the **left–right axis
  from the bicoxofemoral line** (vector between the two femoral-head centres) and
  the cranio-caudal axis from the spine centroid line. The patient **sagittal
  plane normal = the L–R axis.** Project geometry onto this plane for all sagittal
  angles. This is data-derived, so it's robust to patient roll/scan tilt — no
  reliance on the scanner axes being aligned with the patient.

**Fidelity rules of thumb:** prefer surface fits over single-voxel landmarks;
prefer least-squares over min/max extremes (less noise-sensitive); always project
to the patient sagittal/coronal plane (§ above) rather than using image axes;
report every fit's residual as a QC signal.

---

## 4. Output contract (so all miniprojects are comparable)

Every project emits **one JSON record per case** plus an aggregate CSV. Minimum
schema:

```json
{
  "case_id": "0001",
  "parameter": "pelvic_incidence",
  "value": 52.3,
  "units": "degrees",
  "landmarks_world_mm": { "...": [x, y, z] },
  "fit_residuals": { "s1_endplate_rms": 0.6, "femhead_L": 0.4, "femhead_R": 0.5 },
  "qc_flags": ["ok"],
  "method_version": "pi-v1",
  "supine_ct": true
}
```

**Standard `qc_flags`:** `ok`, `fov_truncated`, `missing_label:<id>`,
`low_voxels:<id>`, `fit_residual_high:<which>`, `identity_violation`,
`asymmetry_high`. Downstream QC and validation depend on these — don't silently
drop a bad case, flag it.

---

## 5. The golden rule: position & FOV validity

| Class | Parameters | Why |
|---|---|---|
| **Valid on supine CT** | **PI** (and hip morphology) | PI is a *morphological constant* — posture-invariant, so supine CT = standing. This is the toolbox's strongest, most defensible output. |
| **Supine surrogate (flag it)** | LL, SS, PT, Cobb, wedging, disc/segment spacing, LDI | Computable, clinically meaningful *relatively*, but the absolute value differs from a standing film. Always set `"supine_ct": true` and say so. |
| **Out of scope (v3)** | **SVA, TPA, global coronal balance** | Need C7/T1 and/or standing acquisition — not present in this FOV. Don't fake them. |

---

## 6. Clinical-utility build order (with pre-/post-requirement outlines)

Build top-down. ✅ feasible from v3 · ⚠️ feasible-but-limited · ❌ out of scope.

### Tier 1 — build first

**1. Pelvic Incidence / SS / PT** ✅ — folder `sacral-slope-pelvic-incidence/`
*The flagship. Full worked spec lives in that folder's README.*
- **Pre:** labels S1(7)/sacrum(8), femur_left(11), femur_right(12), hips(9/10); CT affine.
- **Post:** PI, SS, PT (deg) + landmarks + fit residuals; QC `|SS+PT−PI|<1°`.

**2. Lumbar Lordosis (LL)** ✅ — folder `lordosis-trall-angle/`
- **Pre:** L1 superior endplate + S1 superior endplate (ids 1, 7/8); patient sagittal plane.
- **Method:** angle between the two endplate planes, projected to the sagittal plane (Cobb-style). Also emit per-segment lordosis (L1–L2 … L5–S1) for downstream LDI.
- **Post:** LL (deg) + per-segment lordosis; `supine_ct:true`. QC: each endplate fit residual.

**3. PI–LL mismatch** ✅ — derive in the PI or LL project (no new folder needed)
- **Pre:** PI (param 1) + LL (param 2) for the same case.
- **Method:** `mismatch = PI − LL`. Flag the surgical target |PI−LL|>9° (Greenberg).
- **Post:** mismatch (deg) + boolean `pi_ll_mismatch_gt9`. **Highest clinical payload** — it's what surgeons act on.

### Tier 2 — feasible, supine-caveated

**4. Coronal Cobb angle** ✅ — `scoliosis-cobb-angle/`
- **Pre:** vertebral body masks (1–6, 13–25 where in FOV); patient **coronal** plane.
- **Method:** per-vertebra coronal tilt from endplate/long-axis; Cobb = max tilt difference between end vertebrae; auto-pick apex/end vertebrae. **Post:** Cobb (deg), end/apex levels, convex side.

**5. Lordosis Distribution Index (LDI)** ✅ — `lordosis-distribution-index/`
- **Pre:** per-segment lordosis from param 2. **Method:** `(L4–S1 lordosis / total LL)×100`. **Post:** LDI (%).

**6. Vertebral body wedging index** ✅ — `vertebral-body-wedging-index/`
- **Pre:** single vertebral body mask + its sagittal plane. **Method:** anterior vs posterior body height along the sagittal mid-line. **Post:** wedge ratio + angle per level.

**7. Disc spacing** ✅ — `disc-spacing/`
- **Pre:** adjacent endplates (inferior of upper, superior of lower). **Method:** mean inter-endplate gap in the disc region (anterior/mid/posterior). **Post:** disc height(s) per level (the disc itself is unlabeled — measure the bony gap).

**8. Lumbar vertebral spacing** ✅ — `lumbar-vertebral-spacing/`
- **Pre:** vertebral centroids. **Method:** inter-centroid world distances L1→S1. **Post:** spacing per level.

**9. Spondylolisthesis** ✅ — `spondylolisthesis/`
- **Pre:** adjacent vertebral bodies + their endplates; sagittal plane. **Method:** AP offset of the upper body's posterior margin relative to the lower endplate; Meyerding grade from % slip. **Post:** slip mm/%, Meyerding grade, level.

**10. Centroid trajectory / tortuosity** ✅ — `centroid-trajectory-tortuosity/`
- **Pre:** all vertebral centroids in FOV. **Method:** 3D polyline through centroids; tortuosity = path/chord; curvature per level. **Post:** trajectory + tortuosity scalars.

**11. Osteoporosis / trabecular HU** ✅ — `osteoporosis-hu/`
- **Pre:** vertebral body mask **+ the CT** (HU). **Method:** erode the body mask to a central trabecular ROI (avoid cortex/posterior elements), sample mean HU per level. **Post:** mean HU per vertebra (BMD surrogate). QC: ROI voxel count, contrast-phase confound note.

### Tier 3 — limited or out of scope (be honest in the README)

**12. Spinal stenosis** ⚠️ — `spinal-stenosis/`
- v3 has **no canal/thecal-sac/cord label** → only the **bony** canal (vertebral ring inner margin) is derivable. Ship a *bony canal diameter/area* approximation and clearly state it is not soft-tissue stenosis. Full version waits on v4 nerve/canal labels.

**13. Sagittal Vertical Axis (SVA)** ❌ — `sagittal-vertical-axis/`
- Needs the **C7 plumb** + standing acquisition. C7 is out of FOV and CT is supine. Mark out of scope; document the requirement for a future standing-radiograph or whole-spine dataset.

**14. T1 Pelvic Angle (TPA)** ❌ — `t1-pelvic-angle/`
- Needs **T1** (FOV-limited; thoracic GT only ~T8→T12) + femoral heads. Out of scope for the cohort; revisit on the subset where T1 is genuinely imaged.

---

## 7. Validation expectation (for any parameter you ship)

A parameter isn't "done" at "it runs." Validate against manual measurement on a
subset: report **MAE**, **ICC** (auto vs manual), and a **Bland–Altman** plot.
For PI specifically, this is the headline result — it's the parameter most worth
defending quantitatively.
