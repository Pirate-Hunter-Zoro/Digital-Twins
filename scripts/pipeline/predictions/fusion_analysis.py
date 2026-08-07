import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import rankdata
import json

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics
from scripts.pipeline.neighbors.neighbor_scheme import NeighborScheme
from scripts.pipeline.predictions.weighting_strategy import WeightingStrategy
from scripts.shared.plots import (
    plot_receiving_operator_characteristic, 
    bootstrap_roc_band,
    N_BOOTSTRAP
)

ALPHA_GRID = np.linspace(0, 1, 21).reshape(-1,1) # ALPHA 1 recovers p_near; ALPHA 0 recovers 1-p_far
FUSION_SCORE_COLUMNS = ['p_near', 'p_fused', 'p_blend', 'p_stack']
ORDERED_RESULT_COLUMNS = ['score', 'roc_score', 'roc_score_ci_low', 'roc_score_ci_high', 'delta_roc_score', 'delta_roc_score_ci_low', 'delta_roc_score_ci_high', 'auprc', 'brier_score', 'weighted_calibration_error', 'calibration_slope', 'calibration_intercept', 'proportion_risk_score_<0.1', 'proportion_risk_score_>0.9']

def load_predictions() -> pd.DataFrame:
    """Load the predictions data frame that includes all the patients with their risk scores given weighting strategy and weighting scheme

    Returns:
        pd.DataFrame: Resulting information on all patients with their risk scores given different prediction strategies
    """
    return pd.read_csv(Path(os.environ['RESULTS_DIR']) / 'summary_predictions.csv', index_col=0)

def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the given predictions data frame to only have results where the neighbor scheme is nearest or farthest and the weighting scheme is uniform

    Args:
        df (pd.DataFrame): Dataframe containing all predictions

    Returns:
        pd.DataFrame: Filtered data frame
    """
    filtered_by_scheme = df[(df['neighbor_scheme'] == NeighborScheme.NEAREST.name) | (df['neighbor_scheme'] == NeighborScheme.FARTHEST.name)]
    filtered_by_weighting = filtered_by_scheme[filtered_by_scheme['weighting_strategy'] == WeightingStrategy.UNIFORM.name]
    return filtered_by_weighting

def pivot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Given a data frame filtered with patient predictions to only specified neighbor schemes and one withing strategy, fuse the predictions into one row

    Args:
        df (pd.DataFrame): Inputed data frame

    Returns:
        pd.DataFrame: Fused data frame
    """
    # Collapse into one row per patient - for each patient, show the predicted risk scores associated with the nearest and farthest neighbor schemes
    collapsed = df.pivot(index='anchor_patient_id', columns='neighbor_scheme', values='predicted_risk')
    # Obtain the labels for each patient
    labels = df.groupby('anchor_patient_id')['true_label'].first()
    collapsed['true_label'] = labels
    return collapsed.rename(columns={'NEAREST':'p_near', 'FARTHEST':'p_far'})

def fuse_df(pivoted_df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe already pivoted to record near and far risk scores over all patients, create the fused risk score in a new column

    Args:
        pivoted_df (pd.DataFrame): Pivoted data frame

    Returns:
        pd.DataFrame: Resulting data frame with the fused score
    """
    pivoted_df['p_fused'] = 0.5*(pivoted_df['p_near'] + (1 - pivoted_df['p_far']))
    pivoted_df['rank_score'] = pivoted_df['p_near'] - pivoted_df['p_far']
    return pivoted_df

def score_column(fused_df: pd.DataFrame, risk_col: str, mode: str) -> dict:
    """Report all of the scoring metrics given the fused data frame and the specified KNN weighting scheme

    Args:
        fused_df (pd.DataFrame): Fused data frame
        risk_col (str): Label to specify which risk score matters
        mode (str): Used for plot filename

    Returns:
        dict: Metrics for the specified column
    """
    labels = fused_df['true_label'].to_numpy()
    risk_scores = fused_df[risk_col].to_numpy()
    metrics = compute_metrics(labels, risk_scores)
    _, ci_low, ci_high = plot_receiving_operator_characteristic(labels, risk_scores, mode)
    metrics['roc_score_ci_low'] = ci_low
    metrics['roc_score_ci_high'] = ci_high
    return metrics

def blend_oof(fused_df: pd.DataFrame) -> tuple[list[float], pd.DataFrame]:
    """Calculate the blended risk score and add it to the returned data frame

    Args:
        fused_df (pd.DataFrame): Original dataframe

    Returns:
        tuple[list[float], pd.DataFrame]: explored alpha values, paired with same dataframe with blended risk score returned
    """
    labels, p_near, inverted_p_far = fused_df['true_label'].to_numpy(), fused_df['p_near'].to_numpy(), 1 - fused_df['p_far'].to_numpy()
    p_blend = np.full((len(fused_df),), np.nan)
    fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(os.environ['SEED']))
    alphas = []
    for (train_indices, test_indices) in fold_splitter.split(p_near.reshape(-1,1), labels):
        # Iterate over the different partitions of training and testing indices for each fold
        train_blends = ALPHA_GRID * p_near[train_indices].reshape(1,-1) + (1 - ALPHA_GRID) * inverted_p_far[train_indices].reshape(1,-1)
        # (num_alpha, 1) x (1, num_train) -> (num_alpha, num_train), weighted scores over all the alphas
        # Over each alpha value, rank the risk scores
        ranks = rankdata(train_blends, axis=1) # (num_alpha, n_train)
        fold_labels = labels[train_indices]
        positives_mask = fold_labels == 1
        positive_ranks_sums = ranks[:, positives_mask].sum(axis=1) # (num_alpha,) - for each alpha, sum the ranks of all the TRD positive patients
        n_pos = fold_labels.sum()
        n_neg = fold_labels.shape[0] - n_pos
        rocs = (positive_ranks_sums - n_pos*(n_pos+1)/2) / (n_pos * n_neg) # (num_alpha,) - for each alpha, fraction of pos/neg pairings where pos has higher risk score
        
        # Select the winning alpha, and if any tied, take the middle of the tied group
        max_roc = rocs.max()
        tied_positions = np.flatnonzero(rocs >= max_roc-1e-12)
        best_position = tied_positions[len(tied_positions) // 2]
        best_alpha = ALPHA_GRID.ravel()[best_position]
        alphas.append(best_alpha)
        
        # Use picked alpha value on test set
        p_blend[test_indices] = best_alpha * p_near[test_indices] + (1 - best_alpha) * inverted_p_far[test_indices] # (n_patients,) filled in on test patient indices
    
    assert not np.isnan(p_blend).any() # All the folds should have been taken care of - every index was a test index at some point
    fused_df['p_blend'] = p_blend
    return alphas, fused_df # Each fold got a potentially different alpha value to generate the risk scores for that fold's test set

def stack_oof(fused_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply logistic regression to find the best weights for p_near and p_far to find a joint estimate for risk score

    Args:
        fused_df (pd.DataFrame): Original dataframe

    Returns:
        tuple[np.ndarray, pd.DataFrame]: coefficients from inspection fit, and resulting newly modified data frame
    """
    labels, p_near, p_far = fused_df['true_label'].to_numpy(), fused_df['p_near'].to_numpy(), fused_df['p_far'].to_numpy()
    feature_matrix = np.column_stack([p_near, p_far])
    estimator = LogisticRegression()
    fold_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(os.environ['SEED']))
    # Takes care of all the book keeping we had to do in blend_oof
    risk_scores = cross_val_predict(estimator, feature_matrix, labels, cv=fold_splitter, method='predict_proba') # shape (n_patients, 2), where the columns are complements of each other (binary probabilities)
    p_stack = risk_scores[:, 1] # probability for TRD positive
    fused_df['p_stack'] = p_stack
    
    # Fit a separate estimator on the feature matrix and labels, and return its coefficients
    inspector_estimator = LogisticRegression()
    inspector_estimator.fit(feature_matrix, labels)
    return inspector_estimator.coef_, fused_df

def paired_delta_ci(fused_df: pd.DataFrame, baseline_col: str, variant_col: str) -> tuple[float, float, float]:
    """Use bootstrapping to gain a confidence interval on the difference between the AUC scores from the baseline and variant risk score columns

    Args:
        fused_df (pd.DataFrame): Data frame containing all risk score values over all variants
        baseline_col (str): Column containing baseline risk scores
        variant_col (str): Column containing variant risk scores

    Returns:
        tuple[float, float, float]: middle, lower, and upper bounds for difference between two respective AUC scores with 95% confidence interval
    """
    labels = fused_df['true_label'].to_numpy()
    base_risk_scores = fused_df[baseline_col].to_numpy()
    variant_risk_scores = fused_df[variant_col].to_numpy()
    rng = np.random.default_rng(int(os.environ['SEED']))
    # Create bootstrapped index samples - basically samples out of the indices which can repeat, sample size equal to the number of observations
    index_matrix = rng.integers(low=0, high=len(labels), size=(N_BOOTSTRAP, len(labels)))
    _, auc_base_over_bootstraps = bootstrap_roc_band(labels, base_risk_scores, index_matrix)
    _, auc_variant_over_bootstraps = bootstrap_roc_band(labels, variant_risk_scores, index_matrix)
    deltas = auc_variant_over_bootstraps - auc_base_over_bootstraps
    lower, upper = np.nanpercentile(deltas, 2.5), np.nanpercentile(deltas, 97.5)
    delta_point_estimate = roc_auc_score(labels, variant_risk_scores) - roc_auc_score(labels, base_risk_scores)
    return delta_point_estimate, lower, upper

def build_comparison_table(fused_df: pd.DataFrame) -> pd.DataFrame:
    """Given the dataframe full of fused scores, return a report of all confidence intervals and every other scoring characteristic over the scored columns

    Args:
        fused_df (pd.DataFrame): Dataframe with fused scores

    Returns:
        pd.DataFrame: Resulting score report
    """
    results = []
    for col in FUSION_SCORE_COLUMNS:
        result = score_column(fused_df, col, f"fusion_{col}")
        result['score'] = col
        if col != 'p_near':
            delta_point_estimate, delta_low, delta_high = paired_delta_ci(fused_df, 'p_near', col)
        else:
            delta_point_estimate, delta_low, delta_high = np.nan, np.nan, np.nan
        result['delta_roc_score'] = delta_point_estimate
        result['delta_roc_score_ci_low'] = delta_low
        result['delta_roc_score_ci_high'] = delta_high
        results.append(result)
    df = pd.DataFrame(results)[ORDERED_RESULT_COLUMNS]
    df.to_csv(Path(os.environ['RESULTS_DIR']) / 'fusion_comparison.csv', index=False)
    return df

def plot_fusion_roc_overlay(fused_df: pd.DataFrame, comparison_results: pd.DataFrame):
    """Plot overlaying ROC curves for the different fusion analyses

    Args:
        fused_df (pd.DataFrame): Risk scores for all different kinds of 
        comparison_results (pd.DataFrame): Contains all score metrics for the different fusion techniques
    """
    labels = fused_df['true_label'].to_numpy()
    indexed_results = comparison_results.set_index('score')
    fig, ax = plt.subplots()
    for col in FUSION_SCORE_COLUMNS:
        risk_array = fused_df[col].to_numpy()
        false_positives, true_positives, _ = roc_curve(labels, risk_array)
        label = f"{col} ROC: {indexed_results.loc[col, 'roc_score']:.4f}\nROC CI: [{indexed_results.loc[col, 'roc_score_ci_low']:.4f},{indexed_results.loc[col, 'roc_score_ci_high']:.4f}]"
        ax.plot(false_positives, true_positives, label=label)
    ax.plot([0,1], [0,1], linestyle='--', color='gray', label='Chance')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    save_path = Path(os.environ['RESULTS_DIR']) / 'roc_curves/roc_curve_fusion_overlay.png'
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

def main():
    working_df = fuse_df(pivot_df(filter_df(load_predictions())))
    per_fold_alphas, blended_results_df = blend_oof(working_df)
    lr_coefs, lr_results_df = stack_oof(blended_results_df)
    comparison_results = build_comparison_table(lr_results_df)
    plot_fusion_roc_overlay(lr_results_df, comparison_results)
    with open(Path(os.environ['RESULTS_DIR']) / 'fusion_fit_parameters.json', 'w') as f:
        json.dump({'per_fold_alphas': per_fold_alphas, 'lr_coefs': lr_coefs.tolist()}, f, indent=4)

if __name__=="__main__":
    main()