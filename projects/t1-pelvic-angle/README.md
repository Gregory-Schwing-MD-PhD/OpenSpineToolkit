# T1 Pelvic Angle (TPA)

**Miniproject — OpenSpineToolkit**

## Goal
Compute TPA: the angle between the line from the femoral head axis to the T1
centroid and the line from the femoral head axis to the S1 endplate midpoint.

## How CTSpinoPelvic1K v3 helps
TPA needs the **T1 centroid** and the **femoral head axis**. v3 ships **both
femurs as GT** (the femoral head axis) plus the **sacrum** (S1 endplate), so the
*pelvic* half of TPA is fully and directly covered — no manual landmarking.

> **FOV caveat (important):** these are spinopelvic scans, so only the thoracic
> vertebrae *inside the field of view* are labelled — usually down from about
> **T8, not up to T1**. True TPA is therefore computable only on the subset of
> cases whose FOV actually reaches T1. For the rest, report coverage or define a
> clearly-documented surrogate (e.g. the most cranial labelled vertebra) and state
> it explicitly. A good first deliverable is simply *how many cases have T1 in
> view*.

## Your code goes here
Add your code to this folder, then fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
