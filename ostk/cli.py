"""ostk.cli — run spinopelvic measurements over a CTSpinoPelvic1K labels folder.

    python -m ostk pi   --labels labels/ --out pi.csv
    python -m ostk ll   --labels labels/ --out ll.jsonl
    python -m ostk cobb --labels labels/ --out cobb.csv
    python -m ostk all  --labels labels/ --out summary.csv --workers 8

`--labels` is a directory of NIfTI label maps (the dataset's `labels/`); the
case id is the leading token of each filename (`0001_seg.nii.gz` -> `0001`).
Output format is chosen by the `--out` extension (`.csv` or `.jsonl`); with no
`--out` a one-line-per-case summary is printed. Worker functions are top-level
so the run parallelises across processes (`--workers`)."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import List

from .parallel import map_cases


def _json_default(o):
    """Coerce numpy scalars (np.bool_, np.float64 from angle math) to native
    Python types so records round-trip through json cleanly."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _case_id(path: str) -> str:
    b = os.path.basename(path)
    for suf in (".nii.gz", ".nii"):
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    return b.split("_")[0]


def _label_files(labels_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(labels_dir, "*.nii.gz")) +
                   glob.glob(os.path.join(labels_dir, "*.nii")))
    return files


# --- top-level workers (picklable for ProcessPoolExecutor) ------------------

def _run_pi(path: str) -> dict:
    from .io import load_label
    from . import metrics
    lab, aff = load_label(path)
    return metrics.pelvic_incidence_from_label(lab, aff, case_id=_case_id(path)).to_dict()


def _run_ll(path: str) -> dict:
    from .io import load_label
    from . import metrics
    lab, aff = load_label(path)
    return metrics.lumbar_lordosis_from_label(lab, aff, case_id=_case_id(path)).to_dict()


def _run_cobb(path: str) -> dict:
    from .io import load_label
    from . import cobb
    lab, aff = load_label(path)
    return cobb.coronal_cobb_from_label(lab, aff, case_id=_case_id(path)).to_dict()


def _run_all(path: str) -> dict:
    from .io import load_label
    from . import metrics
    lab, aff = load_label(path)
    return metrics.spinopelvic_summary_from_label(lab, aff, case_id=_case_id(path))


_WORKERS = {"pi": _run_pi, "ll": _run_ll, "cobb": _run_cobb, "all": _run_all}


# --- flattening for CSV -----------------------------------------------------

def _flatten(rec: dict, cmd: str) -> dict:
    """One flat row per case for the aggregate CSV (full detail stays in JSONL)."""
    if cmd in ("pi", "ll", "cobb"):
        return {
            "case_id": rec.get("case_id"),
            "parameter": rec.get("parameter"),
            "value": rec.get("value"),
            "units": rec.get("units"),
            "qc_flags": ";".join(rec.get("qc_flags", [])),
            "method_version": rec.get("method_version"),
            "supine_ct": rec.get("supine_ct"),
        }
    mm = rec.get("PI-LL") or {}
    sc = rec.get("schwab") or {}
    return {
        "case_id": rec.get("case_id"),
        "PI": rec.get("PI"), "SS": rec.get("SS"), "PT": rec.get("PT"),
        "LL": rec.get("LL"),
        "PI_minus_LL": mm.get("pi_minus_ll"),
        "within_target_9deg": mm.get("within_target_9deg"),
        "schwab_PI_LL": sc.get("PI-LL"),
        "schwab_PT": sc.get("PT"),
        "ll_increase_needed_deg": sc.get("ll_increase_needed_deg"),
        "qc_flags": ";".join(rec.get("qc_flags", [])),
        "supine_ct": rec.get("supine_ct"),
    }


def _write(records: List[dict], out: str, cmd: str) -> None:
    if out.endswith(".jsonl"):
        with open(out, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, default=_json_default) + "\n")
    elif out.endswith(".csv"):
        rows = [_flatten(r, cmd) for r in records]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        raise SystemExit(f"--out must end in .csv or .jsonl (got {out!r})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ostk", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=sorted(_WORKERS), help="parameter set to compute")
    p.add_argument("--labels", required=True, help="directory of NIfTI label maps")
    p.add_argument("--out", help="output file (.csv or .jsonl); prints if omitted")
    p.add_argument("--workers", type=int, default=None, help="process pool size (1 = serial)")
    a = p.parse_args(argv)

    files = _label_files(a.labels)
    if not files:
        raise SystemExit(f"no .nii/.nii.gz files in {a.labels!r}")

    records = map_cases(_WORKERS[a.command], files, workers=a.workers)

    if a.out:
        _write(records, a.out, a.command)
        ok = sum(1 for r in records if "ok" in r.get("qc_flags", []))
        print(f"{len(records)} cases -> {a.out}  ({ok} clean)")
    else:
        for r in records:
            print(json.dumps(_flatten(r, a.command), default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
