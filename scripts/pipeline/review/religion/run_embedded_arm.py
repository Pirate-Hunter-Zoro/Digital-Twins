"""Fit the religion-free embedded arm and compare it to the control and the published arm.

Usage:
    python -m scripts.pipeline.review.religion.run_embedded_arm

Two contrasts, and the first is the one that answers the question:

  against narrative_control  the parity round's re-render with no content change. Both
                             sides went through the same renderer in the same state, so
                             the delta is attributable to the removed field.
  against published          carries the re-render inside it as well as the field, and is
                             reported because it is the number a reader would otherwise
                             compute for themselves.

Artifacts, in ARTIFACTS_DIR/review/religion/narrative_no_religion/:
  test_predictions.parquet     per-patient held-out probabilities
  metrics.json                 the full metric block per model
  ../embedded_arm_deltas.csv   both contrasts, one row per model per contrast
"""

import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.religion.core import (
    NARRATIVE_ARM,
    compare,
    control_predictions,
    fit_arm,
    published_predictions,
    religion_dir,
)
from scripts.shared.utils import VectorSource


def main():
    arm = fit_arm(NARRATIVE_ARM, VectorSource.EMBEDDED)
    rows = compare(arm, control_predictions(), "narrative_no_religion_vs_control")
    rows += compare(arm, published_predictions(VectorSource.EMBEDDED),
                    "narrative_no_religion_vs_published_embedded")
    frame = pd.DataFrame(rows)
    frame.to_csv(religion_dir() / "embedded_arm_deltas.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
