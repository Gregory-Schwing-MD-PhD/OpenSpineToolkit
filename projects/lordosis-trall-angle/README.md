# Lordosis — TRALL Angle

**Miniproject — OpenSpineToolkit**

## Goal
Quantify lumbar lordosis (e.g. the TRALL / total & segmental lordotic angles)
from vertebral endplate orientations.

## Method (as implemented)
Greenberg Fig. 73.1 measures LL as the **Cobb angle between the L1 superior
endplate and the S1 superior endplate**. v3 ships both (L1 = id 1, S1 = id 7), plus
the **femoral heads (11/12)** which give a data-derived **sagittal plane** to project
the angle into — robust to scan tilt, no reliance on image axes. Per-segment
lordosis (L1–L2 … L5–S1) is emitted for downstream LDI. Where the **T12 endplate**
is in FOV it is a useful optional rostral reference, but the default span is L1→S1.

> Supine surrogate: the Cobb construction is exact, but absolute LL differs from a
> standing film — every record carries `supine_ct: true` (SPEC §5).

## Reference implementation (provided)
Tested code lives in [`ostk`](../../ostk/); [`main.py`](main.py) is a thin wrapper:
```bash
pip install -r ../../requirements.txt
python main.py --labels /path/to/labels --out summary.csv   # full PI/LL/PI-LL/Schwab summary
python main.py --labels /path/to/labels --ll-only           # LL + per-segment only
# equivalently: python -m ostk all  (or  ll)  --labels ...
```
Core: `ostk.metrics.lumbar_lordosis[_from_label]`, `pi_ll_mismatch`,
`schwab_sagittal_modifiers`, `spinopelvic_summary_from_label`.

## Your open task — VALIDATION
Compare LL against manual measurement on a subset (MAE / ICC / Bland–Altman), and
sanity-check PI–LL mismatch + Schwab grades clinically. Put your code + figures
here and fill in:
- **What it does:**
- **How to run** (dependencies + command):
- **Author / team:**
