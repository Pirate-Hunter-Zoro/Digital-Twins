"""Fit the feature arm WITH vital signs and compare it to the published feature arm.

Usage:
    python -m scripts.pipeline.review.parity.run_feature_arm

The cheap half of the parity check: no narrative, no embedding, no GPU. The matrix gains
the three vital-sign columns plus one boolean missingness indicator per vital block, and
the numeric branch gains a median imputer fitted inside each grid-search fold. Everything
else -- the split, the grid, the seed -- is the published configuration.

Artifacts, in ARTIFACTS_DIR/review/parity/feature_vitals/:
  test_predictions.parquet    per-patient held-out probabilities
  metrics.json                the full metric block per model
  ../feature_arm_deltas.csv   paired ROC AUC deltas against the published feature arm
"""

import json

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.parity.core import (
    compare_against_published,
    fit_arm,
    parity_dir,
    published_predictions,
)
from scripts.shared.utils import VectorSource


def main():
    parity = fit_arm("feature_vitals", VectorSource.FEATURE)
    published = published_predictions(VectorSource.FEATURE)
    rows = compare_against_published(parity, published, "feature_vitals_vs_published_feature")
    frame = pd.DataFrame(rows)
    frame.to_csv(parity_dir() / "feature_arm_deltas.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    print(json.dumps(rows, indent=4), flush=True)


if __name__ == "__main__":
    main()
