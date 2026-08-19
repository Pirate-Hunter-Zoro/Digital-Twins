import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from scripts.shared.utils import (
    load_trd_set,
    load_feature_matrix,
    get_AD_mappings
)
from scripts.pipeline.predictions.create_train_test_split import create_train_test_split
from scripts.pipeline.predictions.classical_ml import make_classifier
from scripts.pipeline.predictions.trd_prediction_computation import compute_metrics

OVERLAP_FLOOR = 0.1
OVERLAP_CEILING = 0.9

@dataclass
class EligiblePopulations:
    ref_arm_train_matrix: pd.DataFrame
    ref_arm_train_labels: np.ndarray
    comp_arm_train_matrix: pd.DataFrame
    comp_arm_train_labels: np.ndarray
    # When scoring/testing, we don't need to break patients up by which medication arm they belong to - that will only come into play when testing the models on the patients with the same respective medication arm
    eligible_test_matrix: pd.DataFrame
    eligible_test_labels: np.ndarray
    # In the test patients, flag for each one part of the comparison arm
    test_comparison_flag: np.ndarray
    
def build_eligible_populations(spec_dict: dict) -> EligiblePopulations:
    """Break up the eligible patient population into training and testing populations and keep track of their features and TRD labels

    Args:
        spec_dict (dict): Specifies the reference and comparison arms

    Returns:
        EligiblePopulations: dataclass object containing all relevant information on the population
    """
    train_ids, test_ids = create_train_test_split()
    train_matrix, test_matrix = load_feature_matrix(train_ids), load_feature_matrix(test_ids)
    # Maps ALL patients to their respective medication arm
    mappings = get_AD_mappings()
    train_arms = train_matrix.index.map(mappings)
    train_arms = pd.Series(train_arms)
    test_arms = test_matrix.index.map(mappings)
    test_arms = pd.Series(test_arms)
    
    # Grab the reference and comparison arm markers
    ref_arm, compar_arm = spec_dict['reference_arm'], spec_dict['comparison_arm']
    
    # See which patients are in the reference/comparator arms
    train_keep_mask = train_arms.isin([ref_arm, compar_arm]).to_numpy()
    compar_flag_train = (train_arms == compar_arm).astype(int).to_numpy()
    test_keep_mask = test_arms.isin([ref_arm, compar_arm]).to_numpy()
    compar_flag_test = (test_arms == compar_arm).astype(int).to_numpy()
    
    # Load TRD flags
    trd_patients = load_trd_set()
    
    # Apply filtering
    kept_train_matrix = train_matrix[train_keep_mask]
    kept_compar_flag_train = compar_flag_train[train_keep_mask]
    kept_test_matrix = test_matrix[test_keep_mask]
    kept_compar_flag_test = compar_flag_test[test_keep_mask]
    
    kept_train_y, kept_test_y = np.array([int(id in trd_patients) for id in kept_train_matrix.index]),\
        np.array([int(id in trd_patients) for id in kept_test_matrix.index])
        
    return EligiblePopulations(
        ref_arm_train_matrix=kept_train_matrix[kept_compar_flag_train == 0],
        ref_arm_train_labels=kept_train_y[kept_compar_flag_train == 0],
        comp_arm_train_matrix=kept_train_matrix[kept_compar_flag_train == 1],
        comp_arm_train_labels=kept_train_y[kept_compar_flag_train == 1],
        eligible_test_matrix=kept_test_matrix,
        eligible_test_labels=kept_test_y,
        test_comparison_flag=kept_compar_flag_test
    )
    
def score_counterfactual_risks(population: EligiblePopulations) -> pd.DataFrame:
    """Returns predicted probabilities given each treatment group over the entire population

    Args:
        population (EligiblePopulations): Broken up into treatment groups, and train/test groups

    Returns:
        pd.DataFrame: Resulting risk scores in both scenarios of each medication being taken
    """
    ref_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    ref_pipeline.fit(population.ref_arm_train_matrix, population.ref_arm_train_labels)
    comp_pipeline = make_classifier(LogisticRegression(max_iter=1000))
    comp_pipeline.fit(population.comp_arm_train_matrix, population.comp_arm_train_labels)
    
    # Second column of predictions for positive probability
    ref_world_probs = ref_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    comp_world_probs = comp_pipeline.predict_proba(population.eligible_test_matrix)[:, 1]
    
    return pd.DataFrame(
        {
            'risk_ref': ref_world_probs,
            'risk_comp': comp_world_probs,
            'trd_label': population.eligible_test_labels,
            'is_comparison': population.test_comparison_flag
        }
    ).set_index(population.eligible_test_matrix.index)
    
def grade_arm_models(risk_scores: pd.DataFrame) -> dict:
    """For the risk scores which are gradable (not counterfactuals), grade them

    Args:
        risk_scores (pd.DataFrame): Counterfactual and non-counterfactual risk scores

    Returns:
        dict: Performance metrics for non-counterfactual risk scores
    """
    is_comp_mask = risk_scores['is_comparison'] == 1
    is_ref_mask = ~is_comp_mask
    non_counterfactual_comp_scores = risk_scores['risk_comp'][is_comp_mask]
    non_counterfactual_ref_scores = risk_scores['risk_ref'][is_ref_mask]
    comp_flags = risk_scores['trd_label'][is_comp_mask]
    ref_flags = risk_scores['trd_label'][is_ref_mask]
    
    # Now that we have broken up the non-counterfactual risk scores with their flags, grade them
    comp_scores = compute_metrics(comp_flags, non_counterfactual_comp_scores)
    comp_scores['n_gradable'] = int(is_comp_mask.sum())
    comp_scores['n_events'] = int(comp_flags.sum())
    ref_scores = compute_metrics(ref_flags, non_counterfactual_ref_scores)
    ref_scores['n_gradable'] = int(is_ref_mask.sum())
    ref_scores['n_events'] = int(ref_flags.sum())
    
    # Create weighted calibration slopes and intercepts
    bin_edges = np.linspace(0.0, 1.0, 11)
    binned_predictions_comp = np.digitize(non_counterfactual_comp_scores, bin_edges) - 1
    binned_predictions_ref = np.digitize(non_counterfactual_ref_scores, bin_edges) - 1
    # Any bin assignment greater than the maximum allowed bin gets subtracted by 1 - which would only matter if a prediction were exactly 1.0 - unlikely but we'll be pedantic
    overflow_mask_comp = binned_predictions_comp >= bin_edges.shape[0]-1
    binned_predictions_comp[overflow_mask_comp] = binned_predictions_comp[overflow_mask_comp] - 1
    overflow_mask_ref = binned_predictions_ref >= bin_edges.shape[0]-1
    binned_predictions_ref[overflow_mask_ref] = binned_predictions_ref[overflow_mask_ref] - 1
    
    # In order to weight calibration by bin sizes, we need to compute the bins of risk scores and find each of their counts
    comp_bins = []
    ref_bins = []
    for b_idx in range(bin_edges.shape[0]-1):
        bin_low, bin_high = bin_edges[b_idx], bin_edges[b_idx+1]
        in_bin_comp = binned_predictions_comp == b_idx
        if in_bin_comp.any():
            mean_bin_prediction = non_counterfactual_comp_scores[in_bin_comp].mean()
            observed_fraction = comp_flags[in_bin_comp].mean()
            bin_count = int(in_bin_comp.sum())
            comp_bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "n": bin_count,
                "mean_predicted": mean_bin_prediction,
                "observed_fraction": observed_fraction,
            })
            
        in_bin_ref = binned_predictions_ref == b_idx
        if in_bin_ref.any():
            mean_bin_prediction = non_counterfactual_ref_scores[in_bin_ref].mean()
            observed_fraction = ref_flags[in_bin_ref].mean()
            bin_count = int(in_bin_ref.sum())
            ref_bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "n": bin_count,
                "mean_predicted": mean_bin_prediction,
                "observed_fraction": observed_fraction,
            })  
    ref_scores = ref_scores | count_weighted_slope(pd.DataFrame(ref_bins))
    ref_scores['bins'] = ref_bins
    comp_scores = comp_scores | count_weighted_slope(pd.DataFrame(comp_bins))
    comp_scores['bins'] = comp_bins
    
    return {
        'reference': ref_scores,
        'comparison': comp_scores
    }
    
def count_weighted_slope(bin_table: pd.DataFrame) -> dict:
    """Given the bins that normally go into the slope calculation for a calibration curve, return the slope and intercept when each bin is weighted by its size

    Args:
        bin_table (pd.DataFrame): Calibration bins

    Returns:
        dict: Resulting weighted calibration slope and intercept
    """
    model = LinearRegression()
    model.fit(bin_table['mean_predicted'].to_numpy().reshape(-1,1), bin_table['observed_fraction'], bin_table['n']) # Third argument is sample weight
    return {
        "weighted_cal_slope": float(model.coef_[0]),
        "weighted_cal_intercept": float(model.intercept_)
    }