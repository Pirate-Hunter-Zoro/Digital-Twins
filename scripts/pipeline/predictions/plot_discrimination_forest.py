"""Discrimination forest plot across representations and classifiers.

Two stacked panels:

  (A) All eight representation-by-classifier combinations as a dot-and-whisker
      (forest) plot: ROC AUC point estimate plus its bootstrap percentile 95%
      confidence interval, EMBEDDED grouped above FEATURE.
  (B) The head-to-head test panel A cannot supply: the PAIRED difference in ROC
      AUC between representations, with a paired bootstrap 95% interval. Four
      classifier-matched contrasts (EMBEDDED minus FEATURE, same classifier) plus
      the headline best-versus-best contrast.

Panel B exists because reading panel A for overlap is the weak version of the
comparison. The two representations score the *same* held-out patients, so their
AUCs are positively correlated by construction; comparing marginal intervals for
overlap ignores that correlation and is substantially less powerful than
resampling both score vectors on one set of patient draws. This is the same
argument, and the same machinery, as the fusion analysis in Supplement S7.

This is a presentation-layer script. It refits nothing: panel A reads the
already-written `classical_ml_results_<SOURCE>.json` files, and panel B reads the
per-patient held-out probabilities in `test_predictions_<SOURCE>.parquet`, both
written by scripts.pipeline.predictions.classical_ml. It is therefore cheap to
re-run and cannot drift from the tabulated metrics, which come from the same files.

Outputs:
    RESULTS_DIR/discrimination_forest_EMBEDDED_vs_FEATURE.png
    RESULTS_DIR/discrimination_paired_deltas.json
"""

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from dotenv import load_dotenv
load_dotenv()

from scripts.shared.utils import VectorSource
from scripts.shared.plots import bootstrap_roc_band, N_BOOTSTRAP

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

# Panel B styling: one neutral colour, since a signed difference should not be
# coded as belonging to either representation.
DELTA_COLOR = "#4B3F72"

OUTPUT_NAME = "discrimination_forest_EMBEDDED_vs_FEATURE.png"
DELTA_JSON_NAME = "discrimination_paired_deltas.json"


def signed(value: float) -> str:
    """Format a signed delta with a typographic minus rather than a hyphen.

    Matplotlib's own tick formatter uses U+2212 on the delta axis, so annotations
    built with an ASCII hyphen sit visibly shorter than the axis labels beneath them.

    Args:
        value (float): The delta to format.

    Returns:
        str: The value to three decimals, always signed.
    """
    return f"{value:+.3f}".replace("-", "−")


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


def load_group_predictions(results_dir: Path, source: VectorSource) -> pd.DataFrame:
    """Read one representation's per-patient held-out predicted probabilities.

    Args:
        results_dir (Path): RESULTS_DIR for the active encoder/judge pair.
        source (VectorSource): EMBEDDED or FEATURE.

    Returns:
        pd.DataFrame: one row per held-out patient, carrying patient_id,
            true_label, and one probability column per name in MODEL_ORDER.

    Raises:
        FileNotFoundError: if the prediction table for this representation is absent.
        KeyError: if the file is present but missing an expected classifier column.
    """
    path = results_dir / f"test_predictions_{source.name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot build the paired-difference panel: {path} is missing. "
            "Run scripts.pipeline.predictions.classical_ml first (it writes this "
            "table alongside the metrics JSON)."
        )
    predictions_df = pd.read_parquet(path)
    missing = [m for m in MODEL_ORDER if m not in predictions_df.columns]
    if missing:
        raise KeyError(f"{path} is missing classifier columns {missing}")
    return predictions_df


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


def paired_auc_delta(
    labels: np.ndarray,
    embedded_scores: np.ndarray,
    feature_scores: np.ndarray,
    index_matrix: np.ndarray,
) -> tuple[float, float, float]:
    """Paired bootstrap on the EMBEDDED-minus-FEATURE difference in ROC AUC.

    The same resampled patient indices are applied to both score vectors, so
    every bootstrap draw perturbs the two representations on an identical set of
    patients and the resulting delta distribution retains their correlation. The
    point estimate is the difference of the two full-sample AUCs, not the mean of
    the bootstrap deltas, so it agrees exactly with the tabulated values.

    Args:
        labels (np.ndarray): True TRD flags, shape (n_test,).
        embedded_scores (np.ndarray): Embedded-representation probabilities, shape (n_test,).
        feature_scores (np.ndarray): Feature-representation probabilities, shape (n_test,).
        index_matrix (np.ndarray): Resample indices, shape (N_BOOTSTRAP, n_test).

    Returns:
        tuple[float, float, float]: point estimate, 2.5th percentile, 97.5th
            percentile of the paired difference in ROC AUC.
    """
    _, embedded_aucs = bootstrap_roc_band(labels, embedded_scores, index_matrix)
    _, feature_aucs = bootstrap_roc_band(labels, feature_scores, index_matrix)
    deltas = embedded_aucs - feature_aucs
    low, high = np.nanpercentile(deltas, 2.5), np.nanpercentile(deltas, 97.5)
    point = roc_auc_score(labels, embedded_scores) - roc_auc_score(labels, feature_scores)
    return float(point), float(low), float(high)


def build_delta_rows(
    predictions_by_group: dict[VectorSource, pd.DataFrame],
    results_by_group: dict[VectorSource, dict],
) -> list[dict]:
    """Compute every paired contrast panel B draws.

    Two kinds of contrast, and the distinction matters. The four
    classifier-matched contrasts hold the learner fixed and vary only the
    representation, which is the clean test of the representation itself. The
    headline contrast compares each representation's own best classifier, which is
    the number a reader takes away but is selected post hoc on this same test set,
    so it is reported last and read as descriptive.

    Args:
        predictions_by_group (dict[VectorSource, pd.DataFrame]): per-representation
            per-patient probabilities as returned by load_group_predictions.
        results_by_group (dict[VectorSource, dict]): per-representation metrics,
            used only to pick each side's best classifier.

    Returns:
        list[dict]: one dict per contrast carrying y position, label, delta point
            estimate, interval bounds, formatted text, and a `headline` flag.

    Raises:
        ValueError: if the two prediction tables do not describe the same patients
            in the same order, which would silently invalidate the pairing.
    """
    embedded_df = predictions_by_group[VectorSource.EMBEDDED]
    feature_df = predictions_by_group[VectorSource.FEATURE]
    if not embedded_df["patient_id"].equals(feature_df["patient_id"]):
        raise ValueError(
            "EMBEDDED and FEATURE prediction tables are not row-aligned on "
            "patient_id; a paired test across them would be meaningless."
        )
    labels = embedded_df["true_label"].to_numpy()
    if not np.array_equal(labels, feature_df["true_label"].to_numpy()):
        raise ValueError("EMBEDDED and FEATURE prediction tables disagree on true_label.")

    # One index matrix for every contrast, so the rows of panel B are perturbed on
    # the same patient draws and can be read against each other as well as against zero.
    rng = np.random.default_rng(int(os.environ["SEED"]))
    index_matrix = rng.integers(low=0, high=len(labels), size=(N_BOOTSTRAP, len(labels)))

    contrasts = [
        (MODEL_DISPLAY[m], m, m, False) for m in MODEL_ORDER
    ]
    best_embedded = max(MODEL_ORDER, key=lambda m: results_by_group[VectorSource.EMBEDDED][m]["roc_score"])
    best_feature = max(MODEL_ORDER, key=lambda m: results_by_group[VectorSource.FEATURE][m]["roc_score"])
    # Broken across three lines because the tick-label gutter is sized for
    # "Gradient boosting"; on one line this label runs off the left of the figure.
    contrasts.append((
        f"Best vs best\n({MODEL_DISPLAY[best_embedded].lower()}\nvs {MODEL_DISPLAY[best_feature]})",
        best_embedded,
        best_feature,
        True,
    ))

    delta_rows: list[dict] = []
    y = 0.0
    for label, embedded_model, feature_model, is_headline in contrasts:
        if is_headline:
            y += GROUP_GAP  # visual break before the post-hoc contrast
        point, low, high = paired_auc_delta(
            labels,
            embedded_df[embedded_model].to_numpy(),
            feature_df[feature_model].to_numpy(),
            index_matrix,
        )
        delta_rows.append({
            "y": y,
            "label": label,
            "embedded_model": embedded_model,
            "feature_model": feature_model,
            "delta": point,
            "low": low,
            "high": high,
            "headline": is_headline,
            "text": f"{signed(point)} ({signed(low)} to {signed(high)})",
        })
        y += 1.0
    return delta_rows


def draw_forest_panel(ax, rows: list[dict], scores: np.ndarray, error_bar_lengths: np.ndarray):
    """Draw panel A: AUC with its marginal interval, one row per model/representation.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        rows (list[dict]): Plot-ready rows from build_rows.
        scores (np.ndarray): Point AUCs, shape (n_rows,).
        error_bar_lengths (np.ndarray): [below, above] whisker lengths, shape (2, n_rows).
    """
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
            zorder=3,
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

    # Group headers sit in the blank space above each block of four, carrying the
    # group's own colour. They do the work a legend would, without a legend box
    # covering the bottom rows' whiskers.
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
        "(A) Discrimination by representation and classifier",
        fontsize=13, fontweight="bold", loc="left", pad=10,
    )

    ax.set_xlim(0.595, 0.685)
    ax.set_ylim(max(r["y"] for r in rows) + 0.7, min(r["y"] for r in rows) - 1.1)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_delta_panel(ax, delta_rows: list[dict]):
    """Draw panel B: paired EMBEDDED-minus-FEATURE difference in AUC.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        delta_rows (list[dict]): Contrast rows from build_delta_rows.
    """
    # Zero is the whole question here, so it gets a solid reference line rather
    # than a styled one: an interval covering it is a null result.
    ax.axvline(x=0.0, color="black", linewidth=1.2, zorder=1)

    deltas = np.array([r["delta"] for r in delta_rows])
    whiskers = np.zeros(shape=(2, len(delta_rows)))
    for i, r in enumerate(delta_rows):
        whiskers[0, i] = r["delta"] - r["low"]
        whiskers[1, i] = r["high"] - r["delta"]

    for is_headline, marker in ((False, "D"), (True, "*")):
        idx = [i for i, r in enumerate(delta_rows) if r["headline"] is is_headline]
        if not idx:
            continue
        ax.errorbar(
            deltas[idx],
            [delta_rows[i]["y"] for i in idx],
            xerr=whiskers[:, idx],
            fmt=marker,
            markersize=11 if is_headline else 7,
            color=DELTA_COLOR,
            ecolor=DELTA_COLOR,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            linestyle="none",
            zorder=3,
        )

    for r in delta_rows:
        ax.text(
            1.02, r["y"], r["text"],
            transform=ax.get_yaxis_transform(),
            va="center", ha="left",
            fontsize=11, color="black", clip_on=False,
        )

    ax.set_yticks([r["y"] for r in delta_rows])
    ax.set_yticklabels([r["label"] for r in delta_rows], fontsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("Δ ROC AUC, EMBEDDED − FEATURE (paired bootstrap 95% CI)", fontsize=13)
    ax.set_title(
        "(B) Paired head-to-head difference, same held-out patients",
        fontsize=13, fontweight="bold", loc="left", pad=10,
    )

    # Symmetric limits about zero: an asymmetric window would make a null-centred
    # interval look displaced.
    span = max(
        max(abs(r["low"]) for r in delta_rows),
        max(abs(r["high"]) for r in delta_rows),
    )
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.set_ylim(max(r["y"] for r in delta_rows) + 0.7, min(r["y"] for r in delta_rows) - 0.7)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    """Draw both panels and write the figure plus its paired-delta table into RESULTS_DIR."""
    results_dir = Path(os.environ["RESULTS_DIR"])
    results_by_group = {g: load_group_scores(results_dir, g) for g in GROUP_ORDER}
    predictions_by_group = {g: load_group_predictions(results_dir, g) for g in GROUP_ORDER}

    rows, scores, error_bar_lengths = build_rows(results_by_group)
    delta_rows = build_delta_rows(predictions_by_group, results_by_group)

    # Panel A carries eight rows to panel B's five plus a gap, hence the height
    # split. The figure is deliberately wide and short: it goes into the
    # manuscript at the 6in text width, and a near-square raster there stands
    # 5.9in tall, too tall to share its page with the table it belongs beside,
    # so the rest of that page came out blank. Row labels and the annotation
    # columns sit outside the axes, so height is bought back by pacing the rows
    # tighter rather than by shrinking any text.
    fig, (forest_ax, delta_ax) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(9.6, 6.0),
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.42},
    )
    draw_forest_panel(forest_ax, rows, scores, error_bar_lengths)
    draw_delta_panel(delta_ax, delta_rows)

    # Leave room on the right for both annotation columns and on the left for the
    # bold group headers -- all outside the axes. The horizontal span is wider
    # than it was: at the old 0.30-0.71 the plotted axes used 41% of the canvas
    # and the rest was margin, which is what made the figure near-square. The
    # right edge is set by panel B, whose signed annotations ("+0.028 (+0.017 to
    # +0.039)") are six characters longer than panel A's and are the first thing
    # to run off the canvas if this is pushed further out.
    fig.subplots_adjust(left=0.235, right=0.765, top=0.945, bottom=0.095)

    save_path = results_dir / OUTPUT_NAME
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(str(save_path), dpi=220)
    plt.close(fig)
    print(f"Wrote {save_path}")

    # The paired numbers are cited in the manuscript, so they are written out
    # rather than left readable only off the figure.
    delta_json_path = results_dir / DELTA_JSON_NAME
    with open(delta_json_path, "w") as f:
        json.dump(
            {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": int(os.environ["SEED"]),
                "contrast": "EMBEDDED minus FEATURE",
                "rows": [
                    {
                        "label": r["label"].replace("\n", " "),
                        "embedded_model": r["embedded_model"],
                        "feature_model": r["feature_model"],
                        "delta_roc_score": r["delta"],
                        "delta_roc_score_ci_low": r["low"],
                        "delta_roc_score_ci_high": r["high"],
                        "headline": r["headline"],
                    }
                    for r in delta_rows
                ],
            },
            f,
            indent=4,
        )
    print(f"Wrote {delta_json_path}")
    for r in delta_rows:
        print(f"  {r['label'].replace(chr(10), ' '):58s} {r['text']}")


if __name__ == "__main__":
    main()
