# Sacral Slope / Pelvic Tilt / Pelvic Incidence

**Miniproject — OpenSpineToolkit · Tier 1, build FIRST (highest clinical utility)**

Read the shared contract first: [`../../SPEC.md`](../../SPEC.md) (data, label ids,
geometry primitives, output schema, the position/FOV golden rule).

## Goal
Compute the three spinopelvic parameters from masks alone:
- **PI (pelvic incidence)** — morphological constant; **valid on supine CT** (the
  one major sagittal parameter that transfers from CT to clinic unchanged).
- **SS (sacral slope)** and **PT (pelvic tilt)** — supine surrogates (flag them).
- Identity check: **PI = SS + PT**.

## Why this is the biggest win of v3
PI is defined by the **S1 superior endplate** and the **bicoxofemoral (femoral-head)
axis**. v3 ships **both femurs as GT (ids 11/12)** and a **carved S1 (id 7)** on the
**sacrum (id 8)** — so both landmarks come straight from the labels. No estimation.

---

## Pre-requirements (inputs)
- **Labels:** S1 (7) and/or sacrum (8) for the endplate; femur_left (11),
  femur_right (12); hips (9/10) to localise the acetabular interface.
- **CT affine** (world mm). No CT intensities needed.
- **Shared primitives** (SPEC §3): `fit_plane_tls`, `fit_sphere`,
  `patient_sagittal_plane`, `voxel_to_world`.

## Method outline (be clever, high-fidelity)
1. **Femoral-head centres (bicoxofemoral axis).** For each femur (11/12), isolate
   the femoral *head* as the proximal portion adjacent to the hip/acetabulum
   (9/10) — the femur∩hip neighbourhood gives the cup region. **Sphere-fit** the
   head voxels → centre `C_L`, `C_R` (+ radius, + residual). Sphere fit is the
   clever bit: it's robust to a partially FOV-clipped head and to the non-spherical
   neck. Axis midpoint `M = (C_L + C_R)/2`.
2. **S1 superior endplate.** Take the S1 (7) superior surface — or, if only sacrum
   (8) is present, the cranial articular surface at the L5–S1 gap. Extract the
   surface point cloud (`endplate_surface`) and **fit a plane** → endplate midpoint
   `P` (centroid) and normal `n`. Keep the RMS residual.
3. **Patient sagittal plane.** L–R axis = `C_R − C_L`; project `P`, `M`, and `n`
   into the sagittal plane so all angles are in-plane (robust to scan tilt).
4. **Angles (in the sagittal plane):**
   - **PI** = angle between `n` (endplate perpendicular at `P`) and the line `P→M`.
   - **SS** = angle between the S1 endplate line and the horizontal.
   - **PT** = angle between vertical and the line `P→M`.

## Post-requirements (outputs + QC)
- **Values:** PI, SS, PT (degrees).
- **Landmarks (world mm):** `P`, `C_L`, `C_R`, `M`; endplate normal `n`.
- **Fit residuals:** S1-endplate plane RMS; per-head sphere residual + radius.
- **QC flags (SPEC §4):**
  - `identity_violation` if `|SS + PT − PI| > 1°` (your geometry is wrong).
  - `fit_residual_high:*` if a sphere/plane residual exceeds threshold.
  - `asymmetry_high` if `|radius_L − radius_R|` is large (likely a bad head fit).
  - `fov_truncated` / `missing_label:<id>` / `low_voxels:<id>` as applicable.
- Set `"supine_ct": true`. PI is reported as valid; **label SS/PT as supine.**

## Validation (the headline result)
On a subset with manual radiographic PI: report **MAE**, **ICC** (auto vs manual),
and a **Bland–Altman** plot. PI is the parameter most worth defending numerically —
make this table the centrepiece.

## Reference implementation (provided)
A tested implementation ships in [`ostk`](../../ostk/) — this folder's
[`main.py`](main.py) is a thin wrapper:
```bash
pip install -r ../../requirements.txt
python main.py --labels /path/to/labels --out pi.csv      # or: python -m ostk pi --labels ...
```
Core: `ostk.metrics.pelvic_incidence_from_label` (and `pelvic_incidence(...)` for the
point-cloud core, unit-tested on an analytic phantom with PI=SS+PT).

## Your open task — VALIDATION (the headline result)
The geometry is done; the publishable contribution is proving it. On a subset with
**manual radiographic PI**, report **MAE**, **ICC** (auto vs manual), and a
**Bland–Altman** plot; investigate cases the QC flags catch. Put your validation
code + figures here and fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
