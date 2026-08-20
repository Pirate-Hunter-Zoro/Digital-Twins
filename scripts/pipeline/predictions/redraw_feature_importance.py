"""Redraw the feature-importance panels from saved importances.

`feature_importance.py` computes importances by refitting each best-parameter
classifier, which needs the feature matrix and the model cache and is expensive.
When only the styling of the panels changes, there is no reason to refit
anything: the top-K importances, their signs, and the feature names were already
written to `RESULTS_DIR/feature_importance/feature_importance_summary.json`.

This script reads that summary and calls the same `plot_feature_importance`
helper the pipeline uses, so the redrawn panels match what a full pipeline run
would produce. It is a styling-refresh path, not a second source of truth: if
the summary file is absent it refuses rather than recomputing.

Output: RESULTS_DIR/feature_importance/feature_importance_<model>.png
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.feature_importance import (
    MODEL_NAMES,
    TOP_K,
    plot_feature_importance,
)

SUMMARY_NAME = "feature_importance_summary.json"


def main():
    """Redraw one panel per classifier from the saved importance summary."""
    results_dir = Path(os.environ["RESULTS_DIR"])
    summary_path = results_dir / "feature_importance" / SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} is missing, so there are no saved importances to "
            "redraw. Run scripts.pipeline.predictions.feature_importance "
            "instead -- this script deliberately does not refit models."
        )

    with open(summary_path, "r") as f:
        summary = json.load(f)

    redrawn = []
    for model_name in MODEL_NAMES:
        entries = summary.get(model_name)
        if not entries:
            print(f"  skipping {model_name}: no entries in {SUMMARY_NAME}")
            continue

        # Pass "importance" through EXACTLY as stored -- do not re-apply "sign"
        # to it. The two fields mean different things and are not redundant:
        # "importance" is the raw model attribution, which for logistic
        # regression is a signed coefficient and for the tree models is an
        # unsigned impurity/gain score, while "sign" is separately derived from
        # the univariate Spearman correlation with the risk score. Multiplying
        # them flips the genuinely negative logistic coefficients back to
        # positive and mislabels them as risk-raising.
        importances = np.array([e["importance"] for e in entries], dtype=float)
        signs = np.array([e["sign"] for e in entries], dtype=float)
        feature_names = [e["name"] for e in entries]

        plot_feature_importance(
            importances=importances,
            feature_names=feature_names,
            model_name=model_name,
            top_k=min(TOP_K, len(entries)),
            direction_signs=signs,
        )
        redrawn.append(model_name)

    print(f"Redrew {len(redrawn)} feature-importance panel(s): {', '.join(redrawn)}")
    print(f"  into {results_dir / 'feature_importance'}")


if __name__ == "__main__":
    main()
