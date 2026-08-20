"""Discrimination forest plot across representations and classifiers.

Renders all eight representation-by-classifier combinations as a
dot-and-whisker (forest) plot: ROC AUC point estimate plus its bootstrap
percentile 95% confidence interval, EMBEDDED grouped above FEATURE.

This is a presentation-layer script. It refits nothing and recomputes nothing:
it reads the already-written `classical_ml_results_<SOURCE>.json` files out of
RESULTS_DIR, so it is cheap to re-run and cannot drift from the tabulated
metrics, which come from the same files.

Output: RESULTS_DIR/discrimination_forest_EMBEDDED_vs_FEATURE.png
"""

import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import VectorSource

# Fixed classifier order, identical in both groups. Deliberately NOT sorted by
# AUC: the figure exists so the two representations can be read row-for-row,
# and sorting each group independently would destroy that alignment.
MODEL_ORDER = ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")
MODEL_DISPLAY = {
    "logistic_regression": "Logistic regression",
    "random_forest":       "Random forest",
    "gradient_boosting":   "Gradient boosting",
    "xgboost":             "XGBoost",
}

# EMBEDDED first so it lands on top once the y-axis is inverted.
GROUP_ORDER = (VectorSource.EMBEDDED, VectorSource.FEATURE)
GROUP_STYLE = {
    VectorSource.EMBEDDED: {"color": "steelblue", "marker": "o"},
    VectorSource.FEATURE:  {"color": "darkorange", "marker": "s"},
}

# Blank row inserted between the two groups.
GROUP_GAP = 1.0

OUTPUT_NAME = "discrimination_forest_EMBEDDED_vs_FEATURE.png"


def load_group_scores(results_dir: Path, source: VectorSource) -> dict[str, dict]:
    """Read one representation's classical-ML metrics off disk.

    Args:
        results_dir (Path): RESULTS_DIR for the active encoder/judge pair.
        source (VectorSource): EMBEDDED or FEATURE.

    Returns:
        dict[str, dict]: classifier name -> that classifier's metrics dict,
            which carries at least roc_score, roc_score_ci_low, and
            roc_score_ci_high.

    Raises:
        FileNotFoundError: if the metrics file for this representation is absent.
        KeyError: if the file is present but missing an expected classifier.
    """
    path = results_dir / f"classical_ml_results_{source.name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot build the discrimination forest plot: {path} is missing. "
            "Run scripts.pipeline.predictions.classical_ml first."
        )
    with open(path, "r") as f:
        results = json.load(f)

    missing = [m for m in MODEL_ORDER if m not in results]
    if missing:
        raise KeyError(f"{path} is missing classifiers {missing}")
    return results


def build_rows(results_by_group: dict[VectorSource, dict]) -> tuple[list, np.ndarray, np.ndarray]:
    """Flatten the two representation groups into plot-ready rows.

    Assigns each classifier a y position, leaving GROUP_GAP blank rows between
    groups, and converts each confidence interval into whisker lengths measured
    from the point estimate.

    Args:
        results_by_group (dict[VectorSource, dict]): per-representation metrics
            as returned by load_group_scores.

    Returns:
        tuple: (rows, scores, error_bar_lengths) where rows is a list of dicts
            carrying y position, label, group, and formatted CI text; scores has
            shape (n_rows,) of point AUCs; and error_bar_lengths has shape
            (2, n_rows) of [below, above] whisker lengths. The whiskers are
            asymmetric because these are bootstrap PERCENTILE intervals, not
            point-plus-minus-something; passing a single symmetric halfwidth
            would draw eight subtly wrong intervals.
    """
    rows: list[dict] = []
    y = 0.0
    for group in GROUP_ORDER:
        group_results = results_by_group[group]
        for model_name in MODEL_ORDER:
            metrics = group_results[model_name]
            score = metrics["roc_score"]
            low = metrics["roc_score_ci_low"]
            high = metrics["roc_score_ci_high"]
            rows.append({
                "y": y,
                "group": group,
                "label": MODEL_DISPLAY[model_name],
                "score": score,
                "low": low,
                "high": high,
                "text": f"{score:.3f} ({low:.3f}–{high:.3f})",
            })
            y += 1.0
        y += GROUP_GAP

    scores = np.array([r["score"] for r in rows])
    error_bar_lengths = np.zeros(shape=(2, len(rows)))
    for i, r in enumerate(rows):
        error_bar_lengths[0, i] = r["score"] - r["low"]   # whisker below the dot
        error_bar_lengths[1, i] = r["high"] - r["score"]  # whisker above the dot
    return rows, scores, error_bar_lengths


def main():
    """Draw the forest plot and write it into RESULTS_DIR."""
    results_dir = Path(os.environ["RESULTS_DIR"])
    results_by_group = {g: load_group_scores(results_dir, g) for g in GROUP_ORDER}
    rows, scores, error_bar_lengths = build_rows(results_by_group)

    # Reference line is the best FEATURE model: it is the benchmark the embedded
    # arm is judged against, so whether an embedded interval crosses it is the
    # comparison the figure exists to show. A chance line at 0.5 would be far
    # off-scale and carry no information here.
    feature_best = max(
        results_by_group[VectorSource.FEATURE][m]["roc_score"] for m in MODEL_ORDER
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for group in GROUP_ORDER:
        group_rows = [r for r in rows if r["group"] is group]
        idx = [rows.index(r) for r in group_rows]
        style = GROUP_STYLE[group]
        ax.errorbar(
            scores[idx],
            [r["y"] for r in group_rows],
            xerr=error_bar_lengths[:, idx],
            fmt=style["marker"],
            markersize=7,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            linestyle="none",
            label=group.name,
            zorder=3,
        )

    ax.axvline(
        x=feature_best,
        color="dimgray",
        linestyle="--",
        linewidth=1.8,
        zorder=1,
        label=f"Best rule-based model ({feature_best:.3f})",
    )

    # Per-row numeric annotation, pinned just outside the axes on the right.
    # get_yaxis_transform() reads x in axes coords and y in data coords.
    for r in rows:
        ax.text(
            1.02, r["y"], r["text"],
            transform=ax.get_yaxis_transform(),
            va="center", ha="left",
            fontsize=11, color="black", clip_on=False,
        )

    # Group headers sit in the blank space above each block of four.
    for group in GROUP_ORDER:
        first = min(r["y"] for r in rows if r["group"] is group)
        ax.text(
            -0.005, first - 0.72, group.name,
            transform=ax.get_yaxis_transform(),
            va="center", ha="right",
            fontsize=13, fontweight="bold", color=GROUP_STYLE[group]["color"],
            clip_on=False,
        )

    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("ROC AUC (held-out test set)", fontsize=13)
    ax.set_title(
        "Discrimination by representation and classifier",
        fontsize=14, fontweight="bold", pad=12,
    )

    ax.set_xlim(0.595, 0.685)
    ax.set_ylim(max(r["y"] for r in rows) + 0.7, min(r["y"] for r in rows) - 1.1)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Legend goes BELOW the axes, not inside it. The FEATURE whiskers run most of
    # the panel width, so in-axes placement covers the bottom rows.
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
        fontsize=11, frameon=False,
    )

    # Leave room on the right for the annotation column, on the left for the bold
    # group headers, and at the bottom for the legend -- all outside the axes.
    fig.subplots_adjust(left=0.26, right=0.72, top=0.90, bottom=0.20)

    save_path = results_dir / OUTPUT_NAME
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(str(save_path), dpi=220)
    plt.close(fig)
    print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()
