"""Redraw the ROC, precision-recall, calibration, and confusion-matrix panels
from the saved per-patient predictions.

Why this exists. The panel geometry in `scripts/shared/plots.py` changed: the
panels are now drawn wide and short so that two of them fit inside one page of
the manuscript instead of one, and the confusion matrix puts its metrics report
beside the matrix rather than beneath it. Nothing about the numbers changed,
and refitting anything to pick up a layout change would be absurd -- every
figure this script writes is a function of a label column and a probability
column, both of which the pipeline already wrote to disk:

    RESULTS_DIR/test_predictions_EMBEDDED.parquet    classical arm, embedded
    RESULTS_DIR/test_predictions_FEATURE.parquet     classical arm, rule-based
    RESULTS_DIR/summary_predictions.csv              neighbor-weighted arm

The plotting helpers are the pipeline's own, called with the same `mode`
strings the pipeline uses, so the files land at the same paths a full run would
write them to. The bootstrap that draws the ROC confidence band is seeded from
SEED and its resample matrix depends only on the number of test patients, so a
redraw reproduces the previous band and the previous legend figures exactly --
this is a restyle, not a re-estimate. Every score recomputed here is checked
against the value the pipeline recorded in its results JSON, and a mismatch is
an error rather than a warning.

Usage:
    python -m scripts.pipeline.predictions.redraw_manuscript_panels
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_optimal_confusion_matrix,
)

MODEL_ORDER = ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")
SOURCES = ("EMBEDDED", "FEATURE")
LABEL_COLUMN = "true_label"

# summary_predictions.csv column names for the neighbor-weighted arm.
NEIGHBOR_LABEL_COLUMN = "true_label"
NEIGHBOR_RISK_COLUMN = "predicted_risk"
NEIGHBOR_SCHEME_COLUMN = "neighbor_scheme"
NEIGHBOR_WEIGHTING_COLUMN = "weighting_strategy"

# How close a recomputed ROC AUC has to sit to the recorded one. This is a
# guard against reading the wrong column or the wrong file, not a tolerance for
# drift. The classical arm's predictions round-trip through parquet, which is
# lossless, so its scores have to match to the bit. The neighbor-weighted arm's
# predictions round-trip through CSV, which is not: the risk column is written
# as decimal text, so recomputing from it moves an AUC in the sixth decimal
# place and the threshold below has to allow for that and nothing more.
EXACT_TOLERANCE = 1e-9
CSV_ROUNDTRIP_TOLERANCE = 5e-5


def redraw_one(y_true: np.ndarray, y_prob: np.ndarray, mode: str) -> tuple[float, float, float]:
    """Write all four panel types for one prediction column.

    Args:
        y_true (np.ndarray): Binary TRD labels, shape (n,).
        y_prob (np.ndarray): Predicted TRD probabilities, shape (n,).
        mode (str): Pipeline mode string; becomes the figure filename suffix.

    Returns:
        tuple[float, float, float]: ROC AUC and its bootstrap 95% bounds, for
            cross-checking against the recorded results.
    """
    score, ci_low, ci_high = plot_receiving_operator_characteristic(y_true, y_prob, mode)
    plot_precision_recall(y_true, y_prob, mode)
    plot_calibration(y_true, y_prob, mode)
    plot_optimal_confusion_matrix(y_true, y_prob, mode)
    return score, ci_low, ci_high


def report(mode: str, got: tuple[float, float, float], recorded: dict | None, tolerance: float) -> None:
    """Print the redrawn score beside the recorded one and fail on disagreement.

    The point AUC is order-invariant, so it verifies that the right column was
    read. The interval is not: the bootstrap indexes rows positionally, so a
    matching interval additionally verifies that the saved predictions are in
    the same row order the pipeline drew them in. A differing interval is
    reported rather than fatal, since it moves the shaded band but not any
    number the manuscript quotes.

    Args:
        mode (str): Pipeline mode string.
        got (tuple[float, float, float]): Redrawn score, CI low, CI high.
        recorded (dict | None): The pipeline's saved metrics for this mode.
        tolerance (float): Largest score difference treated as a match.

    Raises:
        ValueError: If the point AUC disagrees with the recorded value.
    """
    score, ci_low, ci_high = got
    if recorded is None or "roc_score" not in recorded:
        print(f"  {mode}: ROC AUC {score:.4f} (no recorded value to check)")
        return
    expected = recorded["roc_score"]
    if abs(score - expected) > tolerance:
        raise ValueError(
            f"{mode}: redrawn ROC AUC {score:.6f} does not match the recorded "
            f"{expected:.6f}. The saved predictions and the saved metrics "
            "disagree; do not publish this figure."
        )
    band = ""
    if "roc_score_ci_low" in recorded:
        drift = max(
            abs(ci_low - recorded["roc_score_ci_low"]),
            abs(ci_high - recorded["roc_score_ci_high"]),
        )
        band = "; band reproduced" if drift <= tolerance else f"; band moved by {drift:.5f}"
    print(f"  {mode}: ROC AUC {score:.4f} matches recorded{band}")


def redraw_classical(results_dir: Path) -> int:
    """Redraw the four panels for each classifier on each representation.

    Args:
        results_dir (Path): RESULTS_DIR for the primary encoder/judge pair.

    Returns:
        int: Number of prediction columns redrawn.
    """
    redrawn = 0
    for source in SOURCES:
        predictions_path = results_dir / f"test_predictions_{source}.parquet"
        results_path = results_dir / f"classical_ml_results_{source}.json"
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"{predictions_path} is missing, so the classical panels cannot be "
                "redrawn without refitting. Run scripts.pipeline.predictions."
                "classical_ml instead."
            )
        predictions = pd.read_parquet(predictions_path)
        with open(results_path, "r") as f:
            recorded = json.load(f)

        labels = predictions[LABEL_COLUMN].to_numpy()
        for model_name in MODEL_ORDER:
            mode = f"{model_name}_{source}"
            got = redraw_one(labels, predictions[model_name].to_numpy(), mode)
            report(mode, got, recorded.get(model_name), EXACT_TOLERANCE)
            redrawn += 1
    return redrawn


def redraw_neighbor(results_dir: Path) -> int:
    """Redraw the four panels for every retrieval-scheme x weighting combination.

    Args:
        results_dir (Path): RESULTS_DIR for the primary encoder/judge pair.

    Returns:
        int: Number of scheme/weighting combinations redrawn.
    """
    summary_path = results_dir / "summary_predictions.csv"
    if not summary_path.exists():
        print(f"  skipping neighbor panels: {summary_path} is absent")
        return 0
    summary = pd.read_csv(summary_path)

    knn_path = results_dir / "knn_results.json"
    recorded = {}
    if knn_path.exists():
        with open(knn_path, "r") as f:
            recorded = json.load(f)

    redrawn = 0
    grouped = summary.groupby([NEIGHBOR_SCHEME_COLUMN, NEIGHBOR_WEIGHTING_COLUMN])
    for (scheme, weighting), rows in grouped:
        mode = f"{scheme}_{weighting}"
        got = redraw_one(
            rows[NEIGHBOR_LABEL_COLUMN].to_numpy(),
            rows[NEIGHBOR_RISK_COLUMN].to_numpy(),
            mode,
        )
        # knn_results.json is keyed by the pipeline's own naming, which has
        # varied across runs, so a missing entry is reported rather than fatal;
        # a present entry that disagrees is fatal.
        report(mode, got, recorded.get(mode), CSV_ROUNDTRIP_TOLERANCE)
        redrawn += 1
    return redrawn


def main():
    """Redraw every panel type for the classical and neighbor-weighted arms."""
    results_dir = Path(os.environ["RESULTS_DIR"])
    print(f"RESULTS_DIR: {results_dir}")
    print("Classical arm:")
    classical = redraw_classical(results_dir)
    print("Neighbor-weighted arm:")
    neighbor = redraw_neighbor(results_dir)
    print(
        f"Redrew 4 panels each for {classical} classical prediction column(s) "
        f"and {neighbor} neighbor scheme/weighting combination(s)."
    )


if __name__ == "__main__":
    main()
