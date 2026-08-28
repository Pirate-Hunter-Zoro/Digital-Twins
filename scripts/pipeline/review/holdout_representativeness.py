"""How representative is the held-out set of the training set?

Review item (2026-08-28): calibration and threshold claims are made on a case-enriched
extract, so the reviewer asked how representative the evaluation sample is. The
manuscript already reports the split's size and TRD counts (Table 3) and asserts in prose
that every per-predictor SMD is small, but it never shows the distributions, so the claim
is unverifiable from the paper. This produces the missing table in the same form as the
TRD-stratified cohort table: one row per characteristic, a Training column, a Held-out
column, and the standardized mean difference between them.

Nothing is refit and nothing is predicted. The split comes from
create_train_test_split, which reads the frozen test_patient_ids.txt, so this describes
exactly the patients the published models were scored on.

Vital signs are INCLUDED here even though load_feature_matrix drops them from the
modelling matrix, because the question is whether the two halves of the cohort look
alike, not what the classifiers were fed.

Artifacts, in ARTIFACTS_DIR/review/holdout_representativeness/:
  holdout_vs_training_full.csv   every column and every level, sorted by |SMD|
  holdout_vs_training_table.md   the curated publication table
  holdout_summary.json           split sizes, TRD rates, and the SMD extremes
"""

import json

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.pipeline.review.paths import review_output_dir
from scripts.shared.feature_display_names import humanize_feature_names
from scripts.shared.utils import load_trd_set

import os
from pathlib import Path

AGE_BINS = [0, 30, 45, 65, np.inf]
AGE_LABELS = ['18-29', '30-44', '45-64', '65+']

# The curated table's rows, in manuscript order. Each entry is
# (section heading or None, column, level spec, display label).
# Level spec is 'median' for a continuous row, 'True' for a boolean row, or a literal
# category level; 'Missing' means the null rate of a categorical column.
CURATED_ROWS = [
    ("Demographics", 'AgeInYears', 'median', "Age (years)"),
    (None, 'age_band', '18-29', "Age band: 18-29"),
    (None, 'age_band', '30-44', "Age band: 30-44"),
    (None, 'age_band', '45-64', "Age band: 45-64"),
    (None, 'age_band', '65+', "Age band: 65+"),
    (None, 'Sex', 'Female', "Sex: Female"),
    (None, 'Race_Ethnicity', 'White or Caucasian', "Race: White/Caucasian"),
    (None, 'Race_Ethnicity', 'Black or African American', "Race: Black/African American"),
    (None, 'Race_Ethnicity', 'American Indian or Alaska Native', "Race: Am. Indian/Alaska Native"),
    (None, 'Race_Ethnicity', 'Missing', "Race: Missing"),
    (None, 'PreferredLanguage', 'English Only', "Language: English only"),
    ("Depression phenotype", 'mdd_recurrence', 'Recurrent', "MDD recurrence: Recurrent"),
    (None, 'mdd_severity', 'Severe', "MDD severity: Severe"),
    (None, 'mdd_severity', 'Moderate', "MDD severity: Moderate"),
    ("Psychiatric and substance comorbidity", 'suicide_flag', 'True', "Suicidality flagged"),
    (None, 'psych_ANXIETY', 'True', "Anxiety disorder"),
    (None, 'psych_SUD', 'True', "Substance use disorder (any)"),
    (None, 'psych_INSOMNIA', 'True', "Insomnia"),
    (None, 'psych_PTSD', 'True', "PTSD"),
    (None, 'sud_Alcohol', 'True', "Alcohol use disorder"),
    ("Medical comorbidity", 'medical_HYPERLIPIDEMIA', 'True', "High cholesterol"),
    (None, 'safety_UNCONTROLLED_HTN', 'True', "Uncontrolled hypertension"),
    ("Vital signs (narrative only)", 'bmi', 'median', "BMI"),
    (None, 'bp_sys', 'median', "Systolic BP"),
    (None, 'bmi', 'Missing', "BMI: not recorded"),
    ("Treatment and utilization", 'polypharmacy_count', 'median', "Active med count"),
    (None, 'num_encounters', 'median', "Encounter count"),
    (None, 'pre_anchor_history_days', 'median', "Pre-anchor history (days)"),
    (None, 'trials_any', 'True', "Prior adequate AD trial (any class)"),
    (None, 'benzo_days_recorded', 'True', "Benzodiazepine days recorded"),
    (None, 'hypnotic_recorded', 'True', "Hypnotic recorded"),
    (None, 'augmentation_occured', 'True', "Augmentation therapy used"),
]


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference between two continuous samples (a minus b).

    Uses the same pooled-SD definition as the cohort table in notebooks/
    cohort_investigation.ipynb, so the numbers are comparable to the published Table 2.

    Args:
        a (pd.Series): First sample, nulls dropped internally.
        b (pd.Series): Second sample.

    Returns:
        float: (mean_a - mean_b) / pooled SD, or 0.0 when the pooled SD vanishes.
    """
    a = a.dropna()
    b = b.dropna()
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def smd_binary(p_a: float, p_b: float) -> float:
    """Standardized mean difference between two proportions (a minus b).

    Args:
        p_a (float): First proportion.
        p_b (float): Second proportion.

    Returns:
        float: (p_a - p_b) / pooled SD, or 0.0 when the pooled SD vanishes.
    """
    pooled_sd = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((p_a - p_b) / pooled_sd)


def median_iqr(series: pd.Series) -> str:
    """Format a continuous column as median (IQR) with the manuscript's precision.

    Args:
        series (pd.Series): Values, nulls dropped internally.

    Returns:
        str: e.g. '55 (38-70)'.
    """
    series = series.dropna()
    return f"{series.median():.0f} ({series.quantile(0.25):.0f}-{series.quantile(0.75):.0f})"


def n_pct(mask: pd.Series) -> str:
    """Format a boolean mask as n (%).

    Args:
        mask (pd.Series): Boolean mask over one split.

    Returns:
        str: e.g. '30,850 (72.5%)'.
    """
    return f"{int(mask.sum()):,} ({mask.mean() * 100:.1f}%)"


def load_cohort() -> pd.DataFrame:
    """Load the full feature table with the derived columns the curated rows need.

    Vitals are kept (unlike load_feature_matrix) and three convenience columns are
    added: an age band, an any-class prior-adequate-trial flag, and recorded flags for
    benzodiazepine days and hypnotics, which the manuscript reports as rates rather than
    as the underlying counts.

    Returns:
        pd.DataFrame: 42,579 rows indexed by patient_id.
    """
    cohort = pd.read_parquet(Path(os.environ['FEATURE_DATAFRAME_PATH']))
    obj_cols = cohort.select_dtypes(include='object').columns
    cohort[obj_cols] = cohort[obj_cols].astype('category')
    cohort['age_band'] = pd.cut(
        cohort['AgeInYears'], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype('category')
    trial_cols = [c for c in cohort.columns if c.startswith('trials_')]
    cohort['trials_any'] = cohort[trial_cols].sum(axis=1) > 0
    cohort['benzo_days_recorded'] = cohort['benzo_days_coverage'] > 0
    cohort['hypnotic_recorded'] = cohort['hypnotics_burden'] > 0
    return cohort


def full_smd_table(cohort: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Every column and level, with its training/held-out summary and SMD.

    Args:
        cohort (pd.DataFrame): Output of load_cohort.
        train_mask (pd.Series): True for training patients.

    Returns:
        pd.DataFrame: One row per (variable, level), sorted by descending |SMD|.
    """
    train = cohort[train_mask]
    test = cohort[~train_mask]
    derived = {'age_band', 'trials_any', 'benzo_days_recorded', 'hypnotic_recorded'}
    rows = []
    for col in cohort.select_dtypes(include='float64').columns:
        rows.append({
            'variable': col,
            'display': humanize_feature_names([col])[0],
            'level': 'median (IQR)',
            'training': median_iqr(train[col]),
            'held_out': median_iqr(test[col]),
            'smd': round(smd_continuous(train[col], test[col]), 4),
            'derived': col in derived,
        })
        if cohort[col].isna().any():
            rows.append({
                'variable': col,
                'display': humanize_feature_names([col])[0],
                'level': 'Missing',
                'training': n_pct(train[col].isna()),
                'held_out': n_pct(test[col].isna()),
                'smd': round(smd_binary(train[col].isna().mean(), test[col].isna().mean()), 4),
                'derived': col in derived,
            })
    for col in cohort.select_dtypes(include='bool').columns:
        rows.append({
            'variable': col,
            'display': humanize_feature_names([col])[0],
            'level': 'True',
            'training': n_pct(train[col]),
            'held_out': n_pct(test[col]),
            'smd': round(smd_binary(train[col].mean(), test[col].mean()), 4),
            'derived': col in derived,
        })
    for col in cohort.select_dtypes(include='category').columns:
        for level in cohort[col].cat.categories:
            rows.append({
                'variable': col,
                'display': humanize_feature_names([f"{col}_{level}"])[0],
                'level': str(level),
                'training': n_pct(train[col].eq(level)),
                'held_out': n_pct(test[col].eq(level)),
                'smd': round(smd_binary(train[col].eq(level).mean(), test[col].eq(level).mean()), 4),
                'derived': col in derived,
            })
        if cohort[col].isna().any():
            rows.append({
                'variable': col,
                'display': humanize_feature_names([col])[0],
                'level': 'Missing',
                'training': n_pct(train[col].isna()),
                'held_out': n_pct(test[col].isna()),
                'smd': round(smd_binary(train[col].isna().mean(), test[col].isna().mean()), 4),
                'derived': col in derived,
            })
    table = pd.DataFrame(rows)
    return table.sort_values('smd', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def curated_markdown(cohort: pd.DataFrame, train_mask: pd.Series) -> str:
    """Render the publication table: manuscript Table 2's rows, split by train/test.

    Args:
        cohort (pd.DataFrame): Output of load_cohort.
        train_mask (pd.Series): True for training patients.

    Returns:
        str: A GitHub-flavoured markdown table.
    """
    train = cohort[train_mask]
    test = cohort[~train_mask]
    lines = [
        "| Characteristic | Training | Held-out | SMD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for section, col, level, label in CURATED_ROWS:
        if section is not None:
            lines.append(f"| **{section}** | | | |")
        if level == 'median':
            training = median_iqr(train[col])
            held_out = median_iqr(test[col])
            smd = smd_continuous(train[col], test[col])
        elif level == 'Missing':
            training = n_pct(train[col].isna())
            held_out = n_pct(test[col].isna())
            smd = smd_binary(train[col].isna().mean(), test[col].isna().mean())
        elif level == 'True':
            training = n_pct(train[col])
            held_out = n_pct(test[col])
            smd = smd_binary(train[col].mean(), test[col].mean())
        else:
            training = n_pct(train[col].eq(level))
            held_out = n_pct(test[col].eq(level))
            smd = smd_binary(train[col].eq(level).mean(), test[col].eq(level).mean())
        lines.append(f"| {label} | {training} | {held_out} | {smd:+.3f} |")
    return "\n".join(lines)


def main():
    save_dir = review_output_dir("holdout_representativeness")
    cohort = load_cohort()
    (train_ids, test_ids) = create_train_test_split()
    train_mask = cohort.index.isin(train_ids)
    trd_ids = load_trd_set()
    trd_mask = cohort.index.isin(trd_ids)

    table = full_smd_table(cohort, pd.Series(train_mask, index=cohort.index))
    table.to_csv(save_dir / "holdout_vs_training_full.csv", index=False)

    markdown = curated_markdown(cohort, pd.Series(train_mask, index=cohort.index))
    (save_dir / "holdout_vs_training_table.md").write_text(markdown + "\n")

    # The reported claim is about the predictors, so the outcome row is summarized
    # separately: the split was stratified, which matches the TRD rate by construction.
    modelled = table[~table['derived']]
    summary = {
        'n_train': int(train_mask.sum()),
        'n_test': int((~train_mask).sum()),
        'n_train_trd': int((train_mask & trd_mask).sum()),
        'n_test_trd': int(((~train_mask) & trd_mask).sum()),
        'trd_rate_train': float(trd_mask[train_mask].mean()),
        'trd_rate_test': float(trd_mask[~train_mask].mean()),
        'max_abs_smd': float(modelled['smd'].abs().max()),
        'max_abs_smd_variable': str(modelled.loc[modelled['smd'].abs().idxmax(), 'display']),
        'max_abs_smd_level': str(modelled.loc[modelled['smd'].abs().idxmax(), 'level']),
        'n_rows_abs_smd_at_least_0.1': int((modelled['smd'].abs() >= 0.1).sum()),
        'n_rows': int(len(modelled)),
    }
    with open(save_dir / "holdout_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    print(json.dumps(summary, indent=4), flush=True)
    print(f"\nWrote 3 artifacts to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
