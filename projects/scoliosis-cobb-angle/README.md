# Scoliosis / Cobb Angle

**Miniproject — OpenSpineToolkit**

## Goal
Measure the coronal Cobb angle from the tilt of the end vertebrae of a curve.

## How CTSpinoPelvic1K v3 helps
The Cobb angle is measured between the most-tilted **end vertebrae**, often
**thoracolumbar** — not purely lumbar. v3 adds **thoracic ground truth**, so you
can identify end vertebrae above L1 instead of being capped at the lumbar spine.

> **FOV caveat:** only the thoracic vertebrae inside the spinopelvic field of view
> are labelled — usually down from about **T8**, not the full T1–T13. So you can
> capture thoracolumbar and lower-thoracic curves; upper-thoracic apices may be
> out of view. (Rib labels — a useful independent level check — are **deferred to
> v4**, so don't rely on them yet.)

## Reuse the tested primitives
The Cobb engine is already in `ostk`: **`ostk.geometry.cobb_angle(normal_a,
normal_b, view_normal)`** projects two endplate normals into a viewing plane and
returns their angle. For scoliosis the `view_normal` is the patient **A–P** axis
(coronal view); per-vertebra endplate normals come from `ostk.masks.endplate_points`
+ `ostk.geometry.fit_plane_tls`. (The lordosis project uses the same `cobb_angle`
with the L–R `view_normal` for the sagittal view — see `ostk.metrics.lumbar_lordosis`.)
Your job: per-vertebra coronal tilt over the FOV vertebrae, then auto-pick the
end/apex vertebrae and report Cobb + convex side.

## Your code goes here
Add your code to this folder, then fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
