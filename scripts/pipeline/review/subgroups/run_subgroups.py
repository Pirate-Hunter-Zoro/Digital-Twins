"""Runner: subgroup performance for both representations, plus the supplement tables.

Usage:
    python -m scripts.pipeline.review.subgroups.run_subgroups

Covers every stratum family in STRATUM_FAMILIES: the two prespecified fairness families
(sex, race), the three further sociodemographic families the review named that this cohort
can actually support (age band, marital status, smoking status) plus religion, and two
clinical families (MDD severity, MDD recurrence) that ask a different question — whether
discrimination is even across illness presentation rather than across people.

Covers all three prediction arms, not just the classifiers: the embedded and
feature-vector classifiers from their prediction parquets, and the neighbour-weighted
predictor from summary_predictions.csv, whose sixteen retrieval-scheme x
weighting-strategy pairs are scored per group. Only nearest retrieval is *contrasted*
across groups — asking whether a deliberately uninformative retrieval scheme is evenly
uninformative across subgroups is not a fairness question, and including the three
negative controls would triple the multiplicity burden on the contrasts that are.

Preferred language is absent by necessity, not oversight: 98.9% of the cohort prefers
English, leaving one estimable level and therefore no contrast.

Artifacts, in ARTIFACTS_DIR/review/subgroups/:
  subgroup_performance.csv    every (arm, group, model) row across all three arms
  subgroup_contrasts.csv      every between-group contrast, with raw and BH-adjusted p
  subgroup_table.md           the fairness-family supplement table, primary model
  subgroup_clinical_table.md  the clinical-family table, primary model
  subgroup_contrast_table.md  the contrast table, all four models, fairness families
  subgroup_summary.json       what survives correction, and what is not estimable
  subgroup_forest.png         primary-model subgroup AUCs with their intervals
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.subgroups.core import (
    ARM_EMBEDDED,
    ARM_FEATURE,
    ARM_KNN,
    CLASSIFIER_ARMS,
    KNN_CONTRAST_MODELS,
    KNN_MODELS,
    MIN_EVENTS,
    MODELS,
    PRIMARY_MODEL,
    STRATUM_FAMILIES,
    benjamini_hochberg,
    load_predictions_with_demographics,
    score_contrasts,
    score_groups,
    subgroup_dir,
)

# The neighbour-weighted arm's headline configuration: nearest retrieval with the
# LLM-judged weighting, which is the arm the main text reports.
PRIMARY_KNN_MODEL = "NEAREST_LLM"
ARMS = (ARM_EMBEDDED, ARM_FEATURE, ARM_KNN)

# One representative model per arm for the primary tables and the forest plot. The full
# grid stays in the CSVs; a table showing 24 models against 30 groups would be unreadable
# and would bury the comparison it exists to make.
PRIMARY_BY_ARM = {
    ARM_EMBEDDED: PRIMARY_MODEL,
    ARM_FEATURE: PRIMARY_MODEL,
    ARM_KNN: PRIMARY_KNN_MODEL,
}
ARM_LABELS = {
    ARM_EMBEDDED: "Embedded (logistic regression)",
    ARM_FEATURE: "Feature vector (logistic regression)",
    ARM_KNN: "Neighbour-weighted (nearest, LLM)",
}

# Families reported as fairness evidence, against families reported as clinical
# heterogeneity. The split is editorial rather than statistical -- the arithmetic is
# identical -- but the two answer different questions and should not share a table.
FAIRNESS_FAMILIES = ('sex', 'race', 'age_band', 'marital_status', 'smoking_status', 'religion')
CLINICAL_FAMILIES = ('mdd_severity', 'mdd_recurrence')

BASE_LABELS = {
    'overall': "All held-out patients",
    'male': "Male",
    'female': "Female",
    'white': "White/Caucasian",
    'non_white': "Non-White (recorded)",
    'race_missing': "Race not recorded",
}

FAMILY_OF_BASE_GROUP = {
    'male': 'sex', 'female': 'sex',
    'white': 'race', 'non_white': 'race', 'race_missing': 'race',
}


def group_label(key: str) -> str:
    """Human-readable name for a group key.

    Args:
        key (str): Either a hand-written key ('male') or a namespaced level
            ('age_band:30-44').

    Returns:
        str: Display label.
    """
    if key in BASE_LABELS:
        return BASE_LABELS[key]
    family, level = key.split(':', 1)
    return f"{dict((f['name'], f['label']) for f in STRATUM_FAMILIES)[family]}: {level}"


def group_family(key: str) -> str:
    """Which stratum family a group key belongs to.

    Args:
        key (str): Group key.

    Returns:
        str: Family name, or 'overall'.
    """
    if key == 'overall':
        return 'overall'
    if key in FAMILY_OF_BASE_GROUP:
        return FAMILY_OF_BASE_GROUP[key]
    return key.split(':', 1)[0]


def performance_table(performance: pd.DataFrame, families: tuple[str, ...]) -> str:
    """Render the primary-model performance table for a set of families.

    Args:
        performance (pd.DataFrame): Concatenated output of score_groups.
        families (tuple[str, ...]): Which families to include, in order.

    Returns:
        str: A GitHub-flavoured markdown table.
    """
    lines = [
        "| Group | n | TRD+ | Arm | ROC AUC (95% CI) | Brier | Calibration slope | Calibration-in-the-large |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    primary = performance[
        [row['model'] == PRIMARY_BY_ARM[row['representation']] for _, row in performance.iterrows()]
    ].copy()
    primary['family'] = primary['group'].map(group_family)
    ordered = ['overall'] if 'overall' in families or families == FAIRNESS_FAMILIES else []
    keys = ordered + [
        key for key in primary['group'].unique()
        if key != 'overall' and group_family(key) in families
    ]
    for key in keys:
        for representation in ARMS:
            row = primary[(primary['group'] == key) & (primary['representation'] == representation)]
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
                f"| {group_label(key)} | {int(row['n']):,} | {int(row['n_events']):,} | "
                f"{ARM_LABELS[representation]} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines)


def contrast_table(contrasts: pd.DataFrame, families: tuple[str, ...]) -> str:
    """Render the contrast table across all four classifiers.

    Args:
        contrasts (pd.DataFrame): Output of score_contrasts with BH columns attached.
        families (tuple[str, ...]): Which families to include.

    Returns:
        str: A GitHub-flavoured markdown table.
    """
    subset = contrasts[contrasts['family'].isin(families)]
    lines = [
        "| Contrast | Arm | Model | ΔROC AUC (95% CI) | p | p (BH) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for _, row in subset.iterrows():
        marker = "**" if row['survives_bh'] else ""
        lines.append(
            f"| {contrast_label(row['contrast'])} | {row['representation'].title()} | "
            f"{row['model'].replace('_', ' ')} | "
            f"{marker}{row['delta_roc']:+.3f} ({row['delta_ci_low']:+.3f} to {row['delta_ci_high']:+.3f}){marker} | "
            f"{row['p_value']:.3f} | {row['p_bh']:.3f} |"
        )
    return "\n".join(lines)


def contrast_label(contrast: str) -> str:
    """Readable name for a contrast key.

    Args:
        contrast (str): e.g. 'male_minus_female' or 'age_band:65+_minus_rest'.

    Returns:
        str: Display label.
    """
    if contrast.endswith('_minus_rest'):
        key = contrast[: -len('_minus_rest')]
        return f"{group_label(key)} vs rest"
    left, right = contrast.split('_minus_')
    return f"{group_label(left)} − {group_label(right)}"


def plot_forest(performance: pd.DataFrame, save_dir):
    """Primary-model subgroup AUCs with their intervals, fairness families only.

    Args:
        performance (pd.DataFrame): Concatenated output of score_groups.
        save_dir (Path): Where to write the PNG.

    Returns:
        Path: The written figure.
    """
    primary = performance[
        [row['model'] == PRIMARY_BY_ARM[row['representation']] and row['estimable']
         for _, row in performance.iterrows()]
    ].copy()
    primary['family'] = primary['group'].map(group_family)
    keys = ['overall'] + [
        key for key in primary['group'].unique()
        if key != 'overall' and group_family(key) in FAIRNESS_FAMILIES
    ]
    figure, ax = plt.subplots(figsize=(6.0, 0.34 * len(keys) + 1.2))
    offsets = {ARM_EMBEDDED: +0.24, ARM_FEATURE: 0.0, ARM_KNN: -0.24}
    colours = {ARM_EMBEDDED: '#1f77b4', ARM_FEATURE: '#d62728', ARM_KNN: '#2ca02c'}
    for representation, offset in offsets.items():
        subset = primary[primary['representation'] == representation]
        positions, scores, lows, highs = [], [], [], []
        for i, key in enumerate(keys):
            row = subset[subset['group'] == key]
            if row.empty:
                continue
            row = row.iloc[0]
            positions.append(len(keys) - 1 - i + offset)
            scores.append(row['roc_score'])
            lows.append(row['roc_score'] - row['roc_ci_low'])
            highs.append(row['roc_ci_high'] - row['roc_score'])
        ax.errorbar(
            scores, positions, xerr=[lows, highs], fmt='o', markersize=4.0,
            capsize=2.0, linewidth=1.0, color=colours[representation],
            label=ARM_LABELS[representation],
        )
    ax.axvline(0.5, color='grey', linewidth=0.8, linestyle=':')
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([group_label(k) for k in reversed(keys)], fontsize=7)
    ax.set_xlabel("ROC AUC (95% CI)", fontsize=8)
    ax.set_title("Subgroup discrimination, one representative model per arm", fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(axis='x', labelsize=7)
    figure.tight_layout()
    save_path = save_dir / "subgroup_forest.png"
    figure.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    return save_path


def main():
    save_dir = subgroup_dir()
    performance_frames, contrast_frames = [], []
    for arm in ARMS:
        frame = load_predictions_with_demographics(arm)
        performance_frames.append(score_groups(frame, arm))
        contrast_frames.append(score_contrasts(frame, arm))
    performance = pd.concat(performance_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)

    # One correction across the whole reported contrast set, not per family: the
    # multiplicity that matters is the number of comparisons a reader is shown.
    contrasts['p_bh'] = benjamini_hochberg(contrasts['p_value'].to_numpy())
    contrasts['survives_bh'] = contrasts['p_bh'] < 0.05

    performance.to_csv(save_dir / "subgroup_performance.csv", index=False)
    contrasts.to_csv(save_dir / "subgroup_contrasts.csv", index=False)
    (save_dir / "subgroup_table.md").write_text(
        performance_table(performance, FAIRNESS_FAMILIES) + "\n")
    (save_dir / "subgroup_clinical_table.md").write_text(
        performance_table(performance, CLINICAL_FAMILIES) + "\n")
    (save_dir / "subgroup_contrast_table.md").write_text(
        contrast_table(contrasts, FAIRNESS_FAMILIES) + "\n")
    figure_path = plot_forest(performance, save_dir)

    nominal = contrasts[contrasts['excludes_zero']]
    surviving = contrasts[contrasts['survives_bh']]
    not_estimable = sorted(set(performance[~performance['estimable']]['group']))
    summary = {
        'arms': list(ARMS),
        'primary_model_by_arm': PRIMARY_BY_ARM,
        'classifier_models': list(MODELS),
        'knn_models': list(KNN_MODELS),
        'knn_models_contrasted': list(KNN_CONTRAST_MODELS),
        'min_events_for_estimability': MIN_EVENTS,
        'families': [f['name'] for f in STRATUM_FAMILIES],
        'groups_not_estimable': not_estimable,
        'n_contrasts': int(len(contrasts)),
        'n_nominally_significant': int(len(nominal)),
        'n_surviving_bh': int(len(surviving)),
        'nominally_significant': nominal[
            ['representation', 'family', 'contrast', 'model', 'delta_roc', 'p_value', 'p_bh']
        ].to_dict(orient='records'),
        'surviving_bh': surviving[
            ['representation', 'family', 'contrast', 'model', 'delta_roc', 'p_value', 'p_bh']
        ].to_dict(orient='records'),
        'smallest_p_bh': float(np.nanmin(contrasts['p_bh'].to_numpy())),
    }
    with open(save_dir / "subgroup_summary.json", 'w') as f:
        json.dump(summary, f, indent=4, default=float)

    print(contrasts[contrasts['excludes_zero']].to_string(index=False), flush=True)
    print("\n" + json.dumps(
        {k: v for k, v in summary.items() if k not in ('nominally_significant', 'surviving_bh')},
        indent=4), flush=True)
    print(f"\nWrote {figure_path} and 6 sibling artifacts to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
