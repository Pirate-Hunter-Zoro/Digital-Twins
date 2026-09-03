"""Refit the feature arm without the Religion column and compare it to the published one.

Usage:
    python -m scripts.pipeline.review.religion.run_feature_arm

The cheap half: no narrative, no embedding, no GPU. The matrix loses one categorical
column -- and with it the unrecorded level that carries the age gradient -- and everything
else is the published configuration: the same split, the same grid, the same seed.

Artifacts, in ARTIFACTS_DIR/review/religion/feature_no_religion/:
  test_predictions.parquet    per-patient held-out probabilities
  metrics.json                the full metric block per model
  ../feature_arm_deltas.csv   paired ROC AUC deltas against the published feature arm
"""

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.religion.core import (
    FEATURE_ARM,
    compare,
    fit_arm,
    published_predictions,
    religion_dir,
)
from scripts.shared.utils import VectorSource


def main():
    arm = fit_arm(FEATURE_ARM, VectorSource.FEATURE)
    published = published_predictions(VectorSource.FEATURE)
    rows = compare(arm, published, "feature_no_religion_vs_published_feature")
    frame = pd.DataFrame(rows)
    frame.to_csv(religion_dir() / "feature_arm_deltas.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
