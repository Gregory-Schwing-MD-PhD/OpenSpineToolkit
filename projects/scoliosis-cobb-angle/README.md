# Scoliosis / Cobb Angle

**Miniproject — OpenSpineToolbox**

## Goal
Measure the coronal Cobb angle from the tilt of the end vertebrae of a curve.

## How CTSpinoPelvic1K v3 helps
The Cobb angle is measured between the most-tilted **end vertebrae**, often thoracolumbar — not purely lumbar. v3 adds **thoracic ground truth**, so you can identify end vertebrae above L1 instead of being capped at the lumbar spine. The **rib** labels give an independent rib-bearing-level check to confirm vertebral identity.

FOV caveat: only the thoracic vertebrae inside the spinopelvic field of view are labelled — usually down from about T8, not the full T1-T13.

## What it does
Two pipelines for automated coronal Cobb angle measurement from CT segmentation masks:

- cobb_angle_analysis_v12.py — Lumbar Cobb angle pipeline for CTSpinoPelvic1K dataset. Uses PCA-based endplate isolation with iterative plane fitting to auto-detect the most-tilted vertebral pair. Outputs per-case CSV with L1-L5, L1-Sacrum, and auto-detected Cobb angles.
- cobb_angle_versefusion_v3.py — Thoracic and lumbar Cobb angle pipeline for VerseFusion dataset. Implements iterative plane-normal endplate isolation, outlier rejection, and min-span enforcement to eliminate spurious pairs.

## How to run

Dependencies:
    pip install numpy nibabel scipy tqdm matplotlib

CTSpinoPelvic1K (lumbar):
    python cobb_angle_analysis_v12.py --hub anonymous-neurips-ED/CTSpinoPelvic1K-Sample
    or with local data:
    python cobb_angle_analysis_v12.py --root /path/to/local/data

VerseFusion (thoracic + lumbar):
    python cobb_angle_versefusion_v3.py --data-dir ~/cobb/versefusion_data/ --output-dir ~/cobb/versefusion_results/ --workers 48

## Author / team
Ashley Schehr — Wayne State University School of Medicine
