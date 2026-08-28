"""Subgroup discrimination and calibration for the published models.

Review item (2026-08-28): the withdrawn fairness claim rested on permuting race and
social-determinant content out of the narrative, which cannot speak to subgroup
performance. The claim is gone from the paper; this supplies the evidence that was missing.

NOTHING IS REFIT. That is the correct design, not a shortcut: the question is how the
models the paper reports behave on subpopulations of the patients they were scored on, so
the per-patient held-out probabilities the pipeline already persisted
(test_predictions_{EMBEDDED,FEATURE}.parquet) are exactly the right input. Refitting per
subgroup would answer a different question -- how well a model trained only on women
predicts women -- and would also destroy comparability with every published number.

Groups are the four the review annotation names: male, female, White/Caucasian, and
non-White. Race is recorded as one field with seven levels and 80.0% of the cohort in one
of them, so a level-by-level breakdown would report six strata too small to estimate; the
majority/minority split is what the data support. Patients with no recorded race are
excluded from the race contrast and reported separately rather than folded into either arm.

Two kinds of interval, and they are not interchangeable:
  within-group   a bootstrap over that group's own patients, giving each subgroup AUC its
                 own uncertainty.
  between-group  an UNPAIRED bootstrap: the two groups are disjoint sets of patients, so
                 each is resampled independently and the difference recomputed. The paired
                 machinery used elsewhere in this paper does not apply here.

Calibration is computed directly rather than through compute_metrics. That helper fits its
calibration line over unweighted bin means, which on a few thousand patients describes the
bin grid more than the model -- the same defect the counterfactual package hit on its
smaller arms. Here the slope is the coefficient of a logistic regression of the outcome on
the logit of the predicted risk, which is the standard definition, and
calibration-in-the-large is the difference between mean predicted risk and observed rate.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.paths import review_output_dir
from scripts.shared.plots import N_BOOTSTRAP
from scripts.shared.utils import VectorSource

ANALYSIS_NAME = "subgroups"

MODELS = ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")

# The stratum families, in reporting order. Each entry names the column the levels come
# from, whether that column lives in the prediction frame or has to be joined from the
# feature table, and how the family is contrasted:
#
#   'direct'       a two-arm family, contrasted as arm A minus arm B. Sex and race are
#                  the two prespecified fairness families and keep this form, which is
#                  what makes their numbers identical to the first pass.
#   'level_vs_rest' a multi-level family, each estimable level contrasted against the
#                  union of the others. Comparing the best level to the worst would
#                  select the contrast on the same data that estimates it.
#
# Preferred language is deliberately absent: 98.9% of the cohort prefers English, so only
# one level is estimable and a one-level family admits no contrast.
STRATUM_FAMILIES = (
    {'name': 'sex', 'label': "Sex", 'column': 'Sex', 'joined': False,
     'kind': 'direct', 'levels': ('male', 'female')},
    {'name': 'race', 'label': "Race/ethnicity", 'column': 'Race_Ethnicity', 'joined': False,
     'kind': 'direct', 'levels': ('white', 'non_white')},
    {'name': 'age_band', 'label': "Age band", 'column': 'age_band', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
    {'name': 'marital_status', 'label': "Marital status", 'column': 'MaritalStatus', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
    {'name': 'smoking_status', 'label': "Smoking status", 'column': 'SmokingStatus', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
    {'name': 'religion', 'label': "Religion", 'column': 'Religion', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
    {'name': 'mdd_severity', 'label': "MDD severity", 'column': 'mdd_severity', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
    {'name': 'mdd_recurrence', 'label': "MDD recurrence", 'column': 'mdd_recurrence', 'joined': True,
     'kind': 'level_vs_rest', 'levels': None},
)

# Columns joined from the feature table for the families above.
JOINED_COLUMNS = ('AgeInYears', 'MaritalStatus', 'SmokingStatus', 'Religion',
                  'mdd_severity', 'mdd_recurrence')

AGE_BINS = [0, 30, 45, 65, float('inf')]
AGE_LABELS = ('18-29', '30-44', '45-64', '65+')
# The model the review annotation asks for; the other three ride along because their
# predictions are already on disk and a one-model answer invites the obvious follow-up.
PRIMARY_MODEL = "logistic_regression"

WHITE_LEVEL = "White or Caucasian"

# A subgroup needs enough events for an AUC to mean anything. Twenty is the threshold the
# cohort investigation already uses for "not estimable", so it is reused here.
MIN_EVENTS = 20


def subgroup_dir() -> Path:
    """Output directory for this analysis.

    Returns:
        Path: ARTIFACTS_DIR/review/subgroups/.
    """
    return review_output_dir(ANALYSIS_NAME)


def load_predictions_with_demographics(source: VectorSource) -> pd.DataFrame:
    """Join the published held-out probabilities to each patient's sex and race.

    Args:
        source (VectorSource): EMBEDDED or FEATURE.

    Returns:
        pd.DataFrame: One row per held-out patient: patient_id, true_label, one column per
            model, plus Sex and Race_Ethnicity.
    """
    predictions = pd.read_parquet(
        Path(os.environ['RESULTS_DIR']) / f"test_predictions_{source.name}.parquet"
    )
    demographics = pd.read_parquet(
        Path(os.environ['FEATURE_DATAFRAME_PATH']),
        columns=['Sex', 'Race_Ethnicity', *JOINED_COLUMNS],
    )
    joined = predictions.merge(
        demographics, left_on='patient_id', right_index=True, how='left', validate='one_to_one'
    )
    if len(joined) != len(predictions):
        raise ValueError("Demographic join changed the row count; the test set and the feature table disagree.")
    joined['age_band'] = pd.cut(
        joined['AgeInYears'], bins=AGE_BINS, labels=list(AGE_LABELS), right=False
    ).astype('object')
    return joined


def family_levels(frame: pd.DataFrame, family: dict) -> list[str]:
    """The level names a family contributes, in a stable order.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.
        family (dict): One entry from STRATUM_FAMILIES.

    Returns:
        list[str]: Group keys, namespaced by family so two families cannot collide on a
            shared level name.
    """
    if family['levels'] is not None:
        return list(family['levels'])
    values = frame[family['column']].dropna().unique().tolist()
    if family['name'] == 'age_band':
        values = [level for level in AGE_LABELS if level in values]
    else:
        values = sorted(str(value) for value in values)
    return [f"{family['name']}:{value}" for value in values]


def group_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The subgroup definitions, as boolean masks over the held-out frame.

    Sex and race keep their hand-written definitions rather than being generated from the
    registry, because their numbers are already published and a majority/minority collapse
    is not something a generic level enumeration would produce.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.

    Returns:
        dict[str, pd.Series]: Group key -> membership mask.
    """
    race = frame['Race_Ethnicity']
    masks = {
        'overall': pd.Series(True, index=frame.index),
        'male': frame['Sex'].eq('Male'),
        'female': frame['Sex'].eq('Female'),
        'white': race.eq(WHITE_LEVEL),
        'non_white': race.notna() & ~race.eq(WHITE_LEVEL),
        'race_missing': race.isna(),
    }
    for family in STRATUM_FAMILIES:
        if family['kind'] != 'level_vs_rest':
            continue
        column = frame[family['column']]
        for key in family_levels(frame, family):
            level = key.split(':', 1)[1]
            masks[key] = column.astype('object').astype(str).eq(level) & column.notna()
    return masks


def calibration(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Calibration slope and calibration-in-the-large.

    Args:
        y_true (np.ndarray): Observed outcomes, shape (n,).
        y_prob (np.ndarray): Predicted risks, shape (n,).

    Returns:
        tuple[float, float]: The logistic recalibration slope (1.0 is perfect), and mean
            predicted risk minus observed rate (0.0 is perfect).
    """
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    recalibration = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    recalibration.fit(logit, y_true)
    return float(recalibration.coef_[0][0]), float(y_prob.mean() - y_true.mean())


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Percentile bootstrap interval for one group's ROC AUC.

    Draws that lose a class entirely yield no AUC and are dropped, which is why the
    percentiles are taken with nanpercentile.

    Args:
        y_true (np.ndarray): Observed outcomes for this group.
        y_prob (np.ndarray): Predicted risks for this group.
        rng (np.random.Generator): Seeded generator.

    Returns:
        tuple[float, float]: 2.5th and 97.5th percentiles.
    """
    n = len(y_true)
    indices = rng.integers(low=0, high=n, size=(N_BOOTSTRAP, n))
    aucs = np.full(N_BOOTSTRAP, np.nan)
    for i in range(N_BOOTSTRAP):
        sampled_true = y_true[indices[i]]
        if sampled_true.min() == sampled_true.max():
            continue
        aucs[i] = roc_auc_score(sampled_true, y_prob[indices[i]])
    return float(np.nanpercentile(aucs, 2.5)), float(np.nanpercentile(aucs, 97.5))


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order.

    The contrast set here is large — every estimable level of every stratum family, at two
    representations and four classifiers — so an unadjusted 5% rule would be expected to
    return several nominally significant results from noise alone. BH controls the false
    discovery rate rather than the family-wise error rate, which is the right trade for a
    screening analysis whose job is to say where to look.

    Args:
        p_values (np.ndarray): Raw two-sided p-values, shape (n,). NaNs pass through.

    Returns:
        np.ndarray: Adjusted values in the same order, shape (n,).
    """
    adjusted = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(~np.isnan(p_values))
    if len(finite) == 0:
        return adjusted
    ordered = finite[np.argsort(p_values[finite])]
    n = len(ordered)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        candidate = p_values[ordered[rank]] * n / (rank + 1)
        running = min(running, candidate)
        adjusted[ordered[rank]] = running
    return adjusted


def bootstrap_p_value(differences: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a difference being zero.

    Twice the smaller tail mass of the bootstrap distribution, floored at the resolution
    the draw count can express so a p-value of exactly zero is never reported.

    Args:
        differences (np.ndarray): Bootstrap differences, shape (N_BOOTSTRAP,), NaNs allowed.

    Returns:
        float: Two-sided p-value, or NaN if no draw survived.
    """
    usable = differences[~np.isnan(differences)]
    if len(usable) == 0:
        return float('nan')
    tail = min((usable <= 0).mean(), (usable >= 0).mean())
    return float(max(2.0 * tail, 1.0 / len(usable)))


def unpaired_auc_difference(
    y_true_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_true_b: np.ndarray,
    y_prob_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap the AUC difference between two disjoint groups (a minus b).

    Each group is resampled independently, because no patient appears in both.

    Args:
        y_true_a (np.ndarray): Group A outcomes.
        y_prob_a (np.ndarray): Group A predicted risks.
        y_true_b (np.ndarray): Group B outcomes.
        y_prob_b (np.ndarray): Group B predicted risks.
        rng (np.random.Generator): Seeded generator.

    Returns:
        tuple[float, float, float, float]: point estimate, 2.5th and 97.5th percentiles,
            and the two-sided bootstrap p-value from the same draws.
    """
    n_a, n_b = len(y_true_a), len(y_true_b)
    indices_a = rng.integers(low=0, high=n_a, size=(N_BOOTSTRAP, n_a))
    indices_b = rng.integers(low=0, high=n_b, size=(N_BOOTSTRAP, n_b))
    differences = np.full(N_BOOTSTRAP, np.nan)
    for i in range(N_BOOTSTRAP):
        true_a, true_b = y_true_a[indices_a[i]], y_true_b[indices_b[i]]
        if true_a.min() == true_a.max() or true_b.min() == true_b.max():
            continue
        differences[i] = (
            roc_auc_score(true_a, y_prob_a[indices_a[i]])
            - roc_auc_score(true_b, y_prob_b[indices_b[i]])
        )
    point = roc_auc_score(y_true_a, y_prob_a) - roc_auc_score(y_true_b, y_prob_b)
    return (
        float(point),
        float(np.nanpercentile(differences, 2.5)),
        float(np.nanpercentile(differences, 97.5)),
        bootstrap_p_value(differences),
    )


def score_groups(frame: pd.DataFrame, source: VectorSource) -> pd.DataFrame:
    """Per-group, per-model discrimination and calibration.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.
        source (VectorSource): Which representation these predictions came from.

    Returns:
        pd.DataFrame: One row per (representation, group, model).
    """
    masks = group_masks(frame)

    def one_cell(group: str, mask: pd.Series, model: str) -> dict:
        """Score one (group, model) cell. Re-seeds from SEED so the result does not
        depend on how the work was distributed across workers."""
        subset = frame[mask]
        y_true = subset['true_label'].to_numpy()
        n_events = int(y_true.sum())
        row = {
            'representation': source.name,
            'group': group,
            'model': model,
            'n': int(len(subset)),
            'n_events': n_events,
            'event_rate': float(y_true.mean()) if len(subset) else np.nan,
        }
        if not estimable(subset['true_label']):
            # Reported as not estimable rather than as a number nobody should read.
            row.update({
                'estimable': False,
                'roc_score': np.nan, 'roc_ci_low': np.nan, 'roc_ci_high': np.nan,
                'brier_score': np.nan, 'calibration_slope': np.nan,
                'calibration_in_the_large': np.nan,
            })
            return row
        rng = np.random.default_rng(int(os.environ['SEED']))
        y_prob = subset[model].to_numpy()
        ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, rng)
        slope, in_the_large = calibration(y_true, y_prob)
        row.update({
            'estimable': True,
            'roc_score': float(roc_auc_score(y_true, y_prob)),
            'roc_ci_low': ci_low,
            'roc_ci_high': ci_high,
            'brier_score': float(brier_score_loss(y_true, y_prob)),
            'calibration_slope': slope,
            'calibration_in_the_large': in_the_large,
        })
        return row

    rows = Parallel(n_jobs=int(os.environ.get('SUBGROUP_JOBS', '1')))(
        delayed(one_cell)(group, mask, model)
        for group, mask in masks.items()
        for model in MODELS
    )
    return pd.DataFrame(rows)


def score_contrasts(frame: pd.DataFrame, source: VectorSource) -> pd.DataFrame:
    """Every between-group AUC contrast the stratum registry defines.

    Two-arm families are contrasted directly, arm A minus arm B. Multi-level families are
    contrasted level against the union of the remaining levels, which avoids selecting the
    comparison on the same data that estimates it — a best-versus-worst contrast over five
    levels would be biased away from zero by construction.

    A contrast is skipped when either side falls below the estimability floor, so a family
    with one usable level contributes nothing rather than a number nobody should read.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.
        source (VectorSource): Which representation these predictions came from.

    Returns:
        pd.DataFrame: One row per (representation, family, contrast, model).
    """
    masks = group_masks(frame)
    jobs = []
    for family in STRATUM_FAMILIES:
        if family['kind'] == 'direct':
            group_a, group_b = family['levels']
            pairs = [(f"{group_a}_minus_{group_b}", masks[group_a], masks[group_b])]
        else:
            pairs = []
            for key in family_levels(frame, family):
                level_mask = masks[key]
                rest_mask = frame[family['column']].notna() & ~level_mask
                pairs.append((f"{key}_minus_rest", level_mask, rest_mask))

        for label, mask_a, mask_b in pairs:
            subset_a, subset_b = frame[mask_a], frame[mask_b]
            if not (estimable(subset_a['true_label']) and estimable(subset_b['true_label'])):
                continue
            for model in MODELS:
                jobs.append((family['name'], label, model, subset_a, subset_b))

    # Every job re-seeds from SEED, so the result does not depend on how the work was
    # split across workers -- the parallelism is a wall-clock change and nothing else.
    def one_contrast(family_name, label, model, subset_a, subset_b):
        rng = np.random.default_rng(int(os.environ['SEED']))
        point, low, high, p_value = unpaired_auc_difference(
            subset_a['true_label'].to_numpy(), subset_a[model].to_numpy(),
            subset_b['true_label'].to_numpy(), subset_b[model].to_numpy(),
            rng,
        )
        return {
            'representation': source.name,
            'family': family_name,
            'contrast': label,
            'model': model,
            'n_a': int(len(subset_a)),
            'n_b': int(len(subset_b)),
            'events_a': int(subset_a['true_label'].sum()),
            'events_b': int(subset_b['true_label'].sum()),
            'delta_roc': point,
            'delta_ci_low': low,
            'delta_ci_high': high,
            'p_value': p_value,
            'excludes_zero': bool(low > 0 or high < 0),
        }

    rows = Parallel(n_jobs=int(os.environ.get('SUBGROUP_JOBS', '1')))(
        delayed(one_contrast)(*job) for job in jobs
    )
    return pd.DataFrame(rows)


def estimable(labels: pd.Series) -> bool:
    """Whether a group has enough of both classes for an AUC to carry meaning.

    Args:
        labels (pd.Series): Outcome column for one group.

    Returns:
        bool: True when both classes clear MIN_EVENTS.
    """
    events = int(labels.sum())
    return events >= MIN_EVENTS and (len(labels) - events) >= MIN_EVENTS
