"""Fit one narrative arm's embedded classifiers and compare to the published embedded arm.

Usage:
    python -m scripts.pipeline.review.parity.run_embedded_arm <arm>

<arm> is 'narrative_control' (re-render, no content change) or 'narrative_parity'
(re-render with pre_anchor_history_days added). Both are compared against the SAME
published embedded predictions, which is what lets the parity delta be split into the part
caused by re-rendering at all and the part caused by the added field.

Artifacts, in ARTIFACTS_DIR/review/parity/<arm>/:
  test_predictions.parquet          per-patient held-out probabilities
  metrics.json                      the full metric block per model
  ../<arm>_deltas.csv               paired ROC AUC deltas against the published arm
"""

import sys

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.parity.core import (
    NARRATIVE_ARMS,
    compare_against_published,
    fit_arm,
    parity_dir,
    published_predictions,
)
from scripts.shared.utils import VectorSource


def main():
    arm = sys.argv[1]
    if arm not in NARRATIVE_ARMS:
        raise ValueError(f"Unknown parity arm {arm!r}; expected one of {sorted(NARRATIVE_ARMS)}.")
    parity = fit_arm(arm, VectorSource.EMBEDDED)
    published = published_predictions(VectorSource.EMBEDDED)
    rows = compare_against_published(parity, published, f"{arm}_vs_published_embedded")
    frame = pd.DataFrame(rows)
    frame.to_csv(parity_dir() / f"{arm}_deltas.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
