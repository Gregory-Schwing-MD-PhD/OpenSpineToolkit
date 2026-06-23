# ostk — OpenSpineToolkit kit

Reusable, **tested** primitives for building spinopelvic measurements from
CTSpinoPelvic1K masks. Functions are pure/stateless → reproducible (no RNG; fixed
sign conventions), low-latency (vectorised + closed-form fits), and picklable for
process-pool parallelism. See the shared contract in [`../SPEC.md`](../SPEC.md).

```bash
pip install -r ../requirements.txt
python -m pytest          # 27 analytic + phantom + CLI tests
```

**Modules**
- `io` — `load_label`, `load_ct`, `voxels_to_world`, `voxel_volume_mm3`
- `geometry` — `fit_sphere`, `fit_plane_tls`, `principal_axes`, `angle_between`, `project_out`, `unit`, `cobb_angle`, `signed_angle_in_plane`, `WORLD_SUPERIOR`
- `masks` — `binary_mask`, `mask_world`, `world_centroid`, `largest_component`, `surface_slab`, `endplate_points`
- `labels` — `LABELS`, `lid()` (the v3/v4 id scheme — no magic numbers)
- `record` — `Measurement` (the per-case output contract)
- `metrics` — Greenberg §73 spinopelvic stack: `pelvic_incidence[_from_label]` (PI/SS/PT), `lumbar_lordosis[_from_label]` (LL + per-segment), `pi_ll_mismatch`, `ll_increase_needed` (Eq. 73.1), `schwab_sagittal_modifiers`, and `spinopelvic_summary_from_label` (everything in one call)
- `cli` — `python -m ostk {pi,ll,all}` batch runner over a `labels/` folder
- `parallel` — `map_cases(fn, items, workers)`

**Run over a dataset (CLI)**
```bash
python -m ostk all --labels labels/ --out summary.csv --workers 8
# per-case detail: --out summary.jsonl
```

**Compose a measurement (library)**
```python
from ostk import load_label, spinopelvic_summary_from_label, map_cases

def summarise(case_id):
    label, affine = load_label(f"labels/{case_id}.nii.gz")
    return spinopelvic_summary_from_label(label, affine, case_id=case_id)

results = map_cases(summarise, case_ids, workers=8)   # PI/SS/PT/LL/PI-LL/Schwab per case
```

**What's implemented (Greenberg §73):** PI/SS/PT, LL (total + per-segment), PI−LL
mismatch + SRS-Schwab modifiers + Eq. 73.1 LL-increase. **Out of scope** (no C7/T1
on supine CT): SVA, TPA. PI is the flagship (valid on supine CT); its absolute
convention + LL are to be confirmed against manual measurement (Paper 2, Aim 2/3).
Geometry cores and the PI=SS+PT / Cobb identities are unit-tested on analytic
phantoms; `_from_label` extraction is the approximate glue pending that validation.
