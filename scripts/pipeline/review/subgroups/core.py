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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.review.paths import review_output_dir
from scripts.shared.plots import N_BOOTSTRAP
from scripts.shared.utils import VectorSource

ANALYSIS_NAME = "subgroups"

MODELS = ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")
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
        Path(os.environ['FEATURE_DATAFRAME_PATH']), columns=['Sex', 'Race_Ethnicity']
    )
    joined = predictions.merge(
        demographics, left_on='patient_id', right_index=True, how='left', validate='one_to_one'
    )
    if len(joined) != len(predictions):
        raise ValueError("Demographic join changed the row count; the test set and the feature table disagree.")
    return joined


def group_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The subgroup definitions, as boolean masks over the held-out frame.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.

    Returns:
        dict[str, pd.Series]: Group label -> membership mask.
    """
    race = frame['Race_Ethnicity']
    return {
        'overall': pd.Series(True, index=frame.index),
        'male': frame['Sex'].eq('Male'),
        'female': frame['Sex'].eq('Female'),
        'white': race.eq(WHITE_LEVEL),
        'non_white': race.notna() & ~race.eq(WHITE_LEVEL),
        'race_missing': race.isna(),
    }


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
        tuple[float, float, float]: point estimate, 2.5th and 97.5th percentiles.
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
    rows = []
    for group, mask in masks.items():
        subset = frame[mask]
        y_true = subset['true_label'].to_numpy()
        n_events = int(y_true.sum())
        for model in MODELS:
            row = {
                'representation': source.name,
                'group': group,
                'model': model,
                'n': int(len(subset)),
                'n_events': n_events,
                'event_rate': float(y_true.mean()) if len(subset) else np.nan,
            }
            if n_events < MIN_EVENTS or (len(subset) - n_events) < MIN_EVENTS:
                # Reported as not estimable rather than as a number nobody should read.
                row.update({
                    'estimable': False,
                    'roc_score': np.nan, 'roc_ci_low': np.nan, 'roc_ci_high': np.nan,
                    'brier_score': np.nan, 'calibration_slope': np.nan,
                    'calibration_in_the_large': np.nan,
                })
            else:
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
            rows.append(row)
    return pd.DataFrame(rows)


def score_contrasts(frame: pd.DataFrame, source: VectorSource) -> pd.DataFrame:
    """The two between-group AUC contrasts the review annotation asks about.

    Args:
        frame (pd.DataFrame): Output of load_predictions_with_demographics.
        source (VectorSource): Which representation these predictions came from.

    Returns:
        pd.DataFrame: One row per (representation, contrast, model).
    """
    masks = group_masks(frame)
    rows = []
    for label, (group_a, group_b) in {
        'male_minus_female': ('male', 'female'),
        'white_minus_non_white': ('white', 'non_white'),
    }.items():
        subset_a, subset_b = frame[masks[group_a]], frame[masks[group_b]]
        for model in MODELS:
            rng = np.random.default_rng(int(os.environ['SEED']))
            point, low, high = unpaired_auc_difference(
                subset_a['true_label'].to_numpy(), subset_a[model].to_numpy(),
                subset_b['true_label'].to_numpy(), subset_b[model].to_numpy(),
                rng,
            )
            rows.append({
                'representation': source.name,
                'contrast': label,
                'model': model,
                'n_a': int(len(subset_a)),
                'n_b': int(len(subset_b)),
                'delta_roc': point,
                'delta_ci_low': low,
                'delta_ci_high': high,
                'excludes_zero': bool(low > 0 or high < 0),
            })
    return pd.DataFrame(rows)
