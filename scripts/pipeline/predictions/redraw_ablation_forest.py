"""Redraw the semantic-feature ablation forest plot from saved artifacts.

Reads `ablation_summary.csv` and `classical_ml_results_EMBEDDED.json` from a
results directory and re-renders `ablation_roc_ci_EMBEDDED.png` in the 2x2
grid the manuscript needs, without refitting or re-embedding anything. Every
recomputed row is checked against the recorded value before the figure is
written.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSIFIER_ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
DISPLAY = {
    "permute_psych_history": "Psychiatric history",
    "permute_med_burden": "Medication burden",
    "permute_treatment_exposure": "Treatment exposure",
    "permute_treatment_contraindications": "Treatment contraindications",
    "permute_sdoh": "Social determinants (SDOH)",
    "permute_race": "Race / ethnicity",
}

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})


def redraw(results_dir: Path) -> Path:
    summary = pd.read_csv(results_dir / "ablation_summary.csv")
    baseline = json.loads((results_dir / "classical_ml_results_EMBEDDED.json").read_text())

    lookup = {(r.spec_id, r.classifier): r for r in summary.itertuples()}

    # Order specs by descending logistic-regression AUC drop, as the caption states.
    drops = sorted(
        ((spec, baseline["logistic_regression"]["roc_score"] - lookup[(spec, "logistic_regression")].roc_score)
         for spec in DISPLAY),
        key=lambda pair: pair[1],
        reverse=True,
    )
    spec_order = [spec for spec, _ in drops]
    labels = ["Baseline"] + [DISPLAY[spec] for spec in spec_order]

    # The delta column and the difference of the two absolute scores must agree.
    for spec in spec_order:
        for clf in CLASSIFIER_ORDER:
            row = lookup[(spec, clf)]
            recomputed = row.roc_score - baseline[clf]["roc_score"]
            if not np.isclose(recomputed, row.delta_roc_score, atol=1e-9):
                raise ValueError(f"delta mismatch for {spec}/{clf}: {recomputed} vs {row.delta_roc_score}")

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10.0, 7.4))
    for i, (clf, ax) in enumerate(zip(CLASSIFIER_ORDER, axes.ravel())):
        mids = np.array([baseline[clf]["roc_score"]] + [lookup[(s, clf)].roc_score for s in spec_order])
        lows = np.array([baseline[clf]["roc_score_ci_low"]] + [lookup[(s, clf)].roc_score_ci_low for s in spec_order])
        highs = np.array([baseline[clf]["roc_score_ci_high"]] + [lookup[(s, clf)].roc_score_ci_high for s in spec_order])
        ax.errorbar(mids, np.arange(len(mids)), xerr=np.array([mids - lows, highs - mids]), fmt="o", capsize=5)
        ax.axvline(baseline[clf]["roc_score"], linestyle="--", linewidth=1, alpha=0.5)
        ax.set_yticks(np.arange(len(mids)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(clf.replace("_", " "))
        if i >= 2:
            ax.set_xlabel("ROC AUC")

    fig.tight_layout()
    out = results_dir / "ablation_roc_ci_EMBEDDED.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="directory holding ablation_summary.csv")
    args = parser.parse_args()
    print(f"wrote {redraw(args.results_dir)}")
