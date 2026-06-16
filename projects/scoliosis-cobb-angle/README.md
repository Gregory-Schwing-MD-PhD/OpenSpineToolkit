# Scoliosis / Cobb Angle

**Miniproject — OpenSpineToolbox**

## Goal
Measure the coronal Cobb angle from the tilt of the end vertebrae of a curve.

## How CTSpinoPelvic1K v3 helps
The Cobb angle is measured between the most-tilted **end vertebrae**, often
**thoracolumbar** — not purely lumbar. v3 adds **thoracic ground truth**, so you
can identify end vertebrae above L1 instead of being capped at the lumbar spine.
The **rib** labels give an independent rib-bearing-level check to confirm
vertebral identity.

> **FOV caveat:** only the thoracic vertebrae inside the spinopelvic field of view
> are labelled — usually down from about **T8**, not the full T1–T13. So you can
> capture thoracolumbar and lower-thoracic curves; upper-thoracic apices may be
> out of view.

## Your code goes here
Add your code to this folder, then fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
