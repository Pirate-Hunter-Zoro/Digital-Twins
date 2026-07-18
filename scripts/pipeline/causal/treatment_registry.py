"""Ordered, append-only registry of pairwise treatment contrasts for the causal-forest sweep.

The SLURM array indexes into TREATMENT_REGISTRY by position. Order is FROZEN once
the sweep has run against it -- only ever append new specs to the end, never
reorder or delete, or a resubmitted array maps indices to the wrong contrast.

Each spec is a single ACTIVE-COMPARATOR contrast: the causal-forest treatment is
which antidepressant CLASS was started at the anchor (index) prescription. For a
given contrast, build_treatment (core.py) assigns:
  - T = 0 to patients whose index class is reference_arm,
  - T = 1 to patients whose index class is comparison_arm,
  - and EXCLUDES every patient whose index class is neither.
Overlap is therefore checked WITHIN the compared pair, not against the whole
cohort. Burden markers and prior-history features stay in X/W as covariates; there
is no per-candidate column drop in this design.

This file is pure data. Each spec declares:
  - key           : short, filename-safe tag; becomes the treatment tag in every
                    metrics JSON and figure filename run_one writes.
  - display_name  : human string for plot titles.
  - reference_arm : the T=0 antidepressant class (a med_definitions class constant).
  - comparison_arm: the T=1 antidepressant class (a med_definitions class constant).

Arm identity comes from the med_definitions constants (the exact strings get_med_arm
returns), so a contrast's arms can never silently mismatch a patient's mapped index
class.
"""

from scripts.data_loading.med_definitions import (
    SSRI,
    SNRI,
    BUPROPION,
)

TREATMENT_REGISTRY = [
    {
        "key": "snri_vs_ssri",
        "display_name": "SNRI vs SSRI",
        "reference_arm": SSRI,
        "comparison_arm": SNRI,
    },
    {
        "key": "bupropion_vs_ssri",
        "display_name": "bupropion vs SSRI",
        "reference_arm": SSRI,
        "comparison_arm": BUPROPION,
    },
    {
        "key": "bupropion_vs_snri",
        "display_name": "bupropion vs SNRI",
        "reference_arm": SNRI,
        "comparison_arm": BUPROPION,
    },
]
