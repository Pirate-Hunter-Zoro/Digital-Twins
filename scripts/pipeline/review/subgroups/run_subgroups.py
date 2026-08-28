"""Runner: subgroup performance for both representations, plus a supplement table.

Usage:
    python -m scripts.pipeline.review.subgroups.run_subgroups

Artifacts, in ARTIFACTS_DIR/review/subgroups/:
  subgroup_performance.csv    every (representation, group, model) row
  subgroup_contrasts.csv      the male-female and White-non-White AUC differences
  subgroup_table.md           the supplement table, primary model only
  subgroup_summary.json       the headline read, and whether any interval excludes zero
  subgroup_forest.png         primary-model subgroup AUCs with their intervals
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.subgroups.core import (
    MIN_EVENTS,
    PRIMARY_MODEL,
    load_predictions_with_demographics,
    score_contrasts,
    score_groups,
    subgroup_dir,
)
from scripts.shared.utils import VectorSource

GROUP_LABELS = {
    'overall': "All held-out patients",
    'male': "Male",
    'female': "Female",
    'white': "White/Caucasian",
    'non_white': "Non-White (recorded)",
    'race_missing': "Race not recorded",
}


def markdown_table(performance: pd.DataFrame) -> str:
    """Render the supplement table for the primary model, both representations.

    Args:
        performance (pd.DataFrame): Concatenated output of score_groups.

    Returns:
        str: A GitHub-flavoured markdown table.
    """
    lines = [
        "| Group | n | TRD+ | Representation | ROC AUC (95% CI) | Brier | Calibration slope | Calibration-in-the-large |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    primary = performance[performance['model'] == PRIMARY_MODEL]
    for group in GROUP_LABELS:
        for representation in ('EMBEDDED', 'FEATURE'):
            row = primary[(primary['group'] == group) & (primary['representation'] == representation)]
            if row.empty:
                continue
            row = row.iloc[0]
            if not row['estimable']:
                cells = ["not estimable", "—", "—", "—"]
            else:
                cells = [
                    f"{row['roc_score']:.3f} ({row['roc_ci_low']:.3f}–{row['roc_ci_high']:.3f})",
                    f"{row['brier_score']:.3f}",
                    f"{row['calibration_slope']:.2f}",
                    f"{row['calibration_in_the_large']:+.3f}",
                ]
            lines.append(
                f"| {GROUP_LABELS[group]} | {int(row['n']):,} | {int(row['n_events']):,} | "
                f"{representation.title()} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def plot_forest(performance: pd.DataFrame, save_dir):
    """Subgroup AUCs with their intervals, primary model, both representations.

    Args:
        performance (pd.DataFrame): Concatenated output of score_groups.
        save_dir (Path): Where to write the PNG.

    Returns:
        Path: The written figure.
    """
    primary = performance[
        (performance['model'] == PRIMARY_MODEL) & performance['estimable']
    ].copy()
    order = [g for g in GROUP_LABELS if g in set(primary['group'])]
    figure, ax = plt.subplots(figsize=(6.0, 3.4))
    offsets = {'EMBEDDED': +0.16, 'FEATURE': -0.16}
    colours = {'EMBEDDED': '#1f77b4', 'FEATURE': '#d62728'}
    for representation, offset in offsets.items():
        subset = primary[primary['representation'] == representation]
        y_positions, scores, lows, highs = [], [], [], []
        for i, group in enumerate(order):
            row = subset[subset['group'] == group]
            if row.empty:
                continue
            row = row.iloc[0]
            y_positions.append(len(order) - 1 - i + offset)
            scores.append(row['roc_score'])
            lows.append(row['roc_score'] - row['roc_ci_low'])
            highs.append(row['roc_ci_high'] - row['roc_score'])
        ax.errorbar(
            scores, y_positions, xerr=[lows, highs], fmt='o', markersize=4.5,
            capsize=2.5, linewidth=1.2, color=colours[representation],
            label=representation.title(),
        )
    ax.axvline(0.5, color='grey', linewidth=0.8, linestyle=':')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in reversed(order)], fontsize=8)
    ax.set_xlabel("ROC AUC (95% CI)", fontsize=9)
    ax.set_title(f"Subgroup discrimination, {PRIMARY_MODEL.replace('_', ' ')}", fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.tick_params(axis='x', labelsize=8)
    figure.tight_layout()
    save_path = save_dir / "subgroup_forest.png"
    figure.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return save_path


def main():
    save_dir = subgroup_dir()
    performance_frames, contrast_frames = [], []
    for source in (VectorSource.EMBEDDED, VectorSource.FEATURE):
        frame = load_predictions_with_demographics(source)
        performance_frames.append(score_groups(frame, source))
        contrast_frames.append(score_contrasts(frame, source))
    performance = pd.concat(performance_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)

    performance.to_csv(save_dir / "subgroup_performance.csv", index=False)
    contrasts.to_csv(save_dir / "subgroup_contrasts.csv", index=False)
    (save_dir / "subgroup_table.md").write_text(markdown_table(performance) + "\n")
    figure_path = plot_forest(performance, save_dir)

    significant = contrasts[contrasts['excludes_zero']]
    not_estimable = performance[~performance['estimable']]['group'].unique().tolist()
    summary = {
        'primary_model': PRIMARY_MODEL,
        'min_events_for_estimability': MIN_EVENTS,
        'groups_not_estimable': not_estimable,
        'n_contrasts': int(len(contrasts)),
        'n_contrasts_excluding_zero': int(len(significant)),
        'contrasts_excluding_zero': significant.to_dict(orient='records'),
        'primary_contrasts': contrasts[contrasts['model'] == PRIMARY_MODEL].to_dict(orient='records'),
        'decision': (
            "at least one subgroup contrast excludes zero; the paper must report it"
            if len(significant) > 0 else
            "no subgroup contrast excludes zero at either representation"
        ),
    }
    with open(save_dir / "subgroup_summary.json", 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(performance[performance['model'] == PRIMARY_MODEL].to_string(index=False), flush=True)
    print("\n" + contrasts[contrasts['model'] == PRIMARY_MODEL].to_string(index=False), flush=True)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k not in ('primary_contrasts', 'contrasts_excluding_zero')}, indent=4), flush=True)
    print(f"\nWrote {figure_path} and 4 sibling artifacts to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
