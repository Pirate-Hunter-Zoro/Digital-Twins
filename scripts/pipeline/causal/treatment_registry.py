"""Ordered, append-only registry of candidate treatments for the causal-forest sweep.

The SLURM array indexes into TREATMENT_REGISTRY by position. Order is FROZEN once
the sweep has run against it -- only ever append new specs to the end, never
reorder or delete, or a resubmitted array maps indices to the wrong treatment.

This file is pure data. Binarization (uniform nonzero-vs-zero) and the
per-candidate confounder drop both live in core.py -- NOT here. Each spec only
declares:
  - key         : short, filename-safe tag; becomes the treatment tag in every
                  metrics JSON and figure filename run_one writes.
  - display_name: human string for plot titles.
  - source_cols : EXACT feature-matrix column names summed to build the binary T.
                  These same names are dropped from X/W for this candidate's run.
"""

TREATMENT_REGISTRY = [
    {
        "key": "polypharmacy",
        "display_name": "Any medications active at index",
        "source_cols": [
            "polypharmacy_count",
        ],
    },
    {
        "key": "trials_bupropion",
        "display_name": "Any bupropion trial",
        "source_cols": [
            "trials_BUPROPION",
        ],
    },
    {
        "key": "trials_mirtazapine",
        "display_name": "Any mirtazapine trial",
        "source_cols": [
            "trials_MIRTAZAPINE",
        ],
    },
    {
        "key": "trials_snri",
        "display_name": "Any SNRI trial",
        "source_cols": [
            "trials_SNRI",
        ],
    },
    {
        "key": "trials_ssri",
        "display_name": "Any SSRI trial",
        "source_cols": [
            "trials_SSRI",
        ],
    },
    {
        "key": "trials_vortioxetine",
        "display_name": "Any vortioxetine trial",
        "source_cols": [
            "trials_VORTIOXETINE",
        ],
    },
    {
        "key": "adequate_trial",
        "display_name": "At least one adequate medication trial (any arm)",
        "source_cols": [
            "trials_BUPROPION",
            "trials_MIRTAZAPINE",
            "trials_SNRI",
            "trials_SSRI",
            "trials_VORTIOXETINE",
        ],
    },
    {
        "key": "augmentation",
        "display_name": "Augmentation occurred",
        "source_cols": [
            "augmentation_occured",
        ],
    },
    {
        "key": "benzo",
        "display_name": "Any benzodiazepine coverage",
        "source_cols": [
            "benzo_days_coverage",
        ],
    },
    {
        "key": "hypnotics",
        "display_name": "Any hypnotics burden",
        "source_cols": [
            "hypnotics_burden",
        ],
    },
    {
        "key": "nsaid",
        "display_name": "Any NSAID use",
        "source_cols": [
            "nsaid_count",
        ],
    },
]
