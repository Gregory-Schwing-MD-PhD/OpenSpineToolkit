"""ostk.labels — the CTSpinoPelvic1K label-id scheme (mirrors the dataset).

v3 populates 0–49 (cores + femurs + GT thoracic + TS ribs) with ignore 50, and
RESERVES the v4 soft-tissue block: iliolumbar (51/52), LS-nerve roots (53–58),
psoas (59/60). v4 populates that block and relocates ignore 50 -> 255. Use the
names so measurement code never hard-codes magic numbers.
"""
from __future__ import annotations

LABELS = {
    "background": 0,
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
    "S1": 7, "sacrum": 8, "left_hip": 9, "right_hip": 10,
    "femur_left": 11, "femur_right": 12,
    **{f"T{n}": 12 + n for n in range(1, 14)},          # T1..T13 -> 13..25
    **{f"rib_left_{n}": 25 + n for n in range(1, 13)},  # 26..37
    **{f"rib_right_{n}": 37 + n for n in range(1, 13)},  # 38..49
    "iliolumbar_left": 51, "iliolumbar_right": 52,
    "nerve_L4_left": 53, "nerve_L4_right": 54,
    "nerve_L5_left": 55, "nerve_L5_right": 56,
    "nerve_S1_left": 57, "nerve_S1_right": 58,
    "psoas_left": 59, "psoas_right": 60,                 # v4 (XLIF corridor)
}
ID_TO_NAME = {v: k for k, v in LABELS.items()}

LUMBAR = ("L1", "L2", "L3", "L4", "L5", "L6")
THORACIC = tuple(f"T{n}" for n in range(1, 14))
IGNORE_V3 = 50
IGNORE_V4 = 255


def lid(name: str) -> int:
    """Label id for a structure name (raises on typo — fail loud, not silent)."""
    return LABELS[name]
