"""Reference implementation -- coronal Cobb angle from CTSpinoPelvic1K v3 masks.

Thin wrapper over the tested `ostk` primitives (fit_endplate + femoral-head-
derived patient coronal axis). Run over a labels folder:

    python main.py --labels /path/to/labels --out cobb.csv

Equivalent to `python -m ostk cobb --labels ... --out ...`. The open student
task here is VALIDATION: compare `value` against manual Cobb measurement on a
subset and report MAE / ICC / Bland-Altman (SPEC Section 7), and cross-check
against the earlier cobb_angle_analysis_v12.py implementation on the same cases.
"""
import argparse

from ostk.cli import main as ostk_main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True, help="directory of NIfTI label maps")
    ap.add_argument("--out", help="output .csv or .jsonl (prints if omitted)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    argv = ["cobb", "--labels", a.labels]
    if a.out:
        argv += ["--out", a.out]
    if a.workers is not None:
        argv += ["--workers", str(a.workers)]
    return ostk_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
