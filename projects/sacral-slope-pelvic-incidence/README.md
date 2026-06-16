# Sacral Slope / Pelvic Incidence

**Miniproject — OpenSpineToolbox**

## Goal
Compute the spinopelvic parameters: sacral slope (SS), pelvic tilt (PT), and
pelvic incidence (PI = SS + PT).

## How CTSpinoPelvic1K v3 helps
This is the **biggest single win** of v3. PI is defined by the relationship
between the **S1 superior endplate** and the **femoral head axis**. v3 ships
**both femurs as GT**, so you can compute the femoral head centres directly (the
bicoxofemoral axis) instead of estimating them — and the **sacrum GT** gives the
S1 endplate for the slope. SS and PT then follow from the labels alone.

> Note: an experimental S1 sub-label may be present in some v3 cases, but you do
> **not** need it — the S1 superior endplate is the cranial surface of the sacrum.

## Your code goes here
Add your code to this folder, then fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
