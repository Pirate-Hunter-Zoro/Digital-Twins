import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    precision_recall_curve, 
    auc,
)
from sklearn.linear_model import LinearRegression

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.predictions.weighting_strategy import WeightingStrategy
from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_effective_sample_size_distribution,
    plot_optimal_confusion_matrix
)
from scripts.shared.utils import load_neighborhood_data

def calculated_weighted_risk(group: pd.DataFrame, strategy: WeightingStrategy) -> tuple[float, float]:
    """Method to calculate the weighted risk of TRD of a patient along with their essential sample size

    Args:
        group (pd.DataFrame): Patient neighborhood information
        strategy (WeightingStrategy): e.g. uniform weighting, cosine weighting, combined weighting

    Raises:
        ValueError: In the case of a patient's neighbors all having zero weight

    Returns:
        tuple[float, float]: TRD risk score paired with effective sample size
    """
    patient_id = group['anchor_patient_id'].iloc[0]
    alpha = float(os.environ['WEIGHTING_EXPONENT'])
    if strategy == WeightingStrategy.LLM or strategy == WeightingStrategy.COMBINED:
        # We cannot work with records that did not have proper LLM judgements
        cleaned_group = group[group['llm_sim'].notna()]
    else:
        cleaned_group = group
        
    # Grab neighbor TRD statuses
    trds = np.array([status for status in cleaned_group['neighbor_trd_label']])
    
    # Assign weights
    if strategy == WeightingStrategy.LLM:
        weights = np.array([(score/100)**alpha for score in cleaned_group['llm_sim']])
    elif strategy == WeightingStrategy.COSINE:
        weights = np.array([max(0,score)**alpha for score in cleaned_group['cosine_sim']])
    elif strategy == WeightingStrategy.COMBINED:
        weights = np.array([(2*max(cos_score,0)*(llm_score/100)/(max(cos_score, 0)+(llm_score/100)+1e-9))**alpha  for llm_score, cos_score in zip(cleaned_group['llm_sim'], cleaned_group['cosine_sim'])])
    else:
        weights = np.ones(shape=(len(cleaned_group),))
    
    # Compute risk by dotting weights with flags and dividing by weights
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        raise ValueError(f"ERROR: all weights zero for patient with ID: {patient_id} using strategy {strategy.name}...")
    risk_score = np.dot(weights, trds) / weight_sum
    
    # Compute effective sample size
    ess = np.sum(weights)**2 / np.sum(weights**2)
    
    return (risk_score, ess)

def compute_metrics(y_true: np.array, y_prob: np.array) -> dict:
    """Compute the metrics to describe the performance of the TRD risk predictions given the actual flags

    Args:
        y_true (np.array): Actual TRD flags of the patients
        y_prob (np.array): Predicted TRD risks for the patients

    Returns:
        dict: Performance results
    """
    # ROC score
    roc_score = roc_auc_score(y_true=y_true, y_score=y_prob)
    
    # PR area curve
    precision, recall, _ = precision_recall_curve(y_true=y_true, y_score=y_prob)
    auprc = auc(x=recall, y=precision)
    
    # Brier score
    brier_score = brier_score_loss(y_true=y_true, y_proba=y_prob)
    
    num_bins = 10
    bin_edges = np.linspace(0.0, 1.0, num_bins+1)
    # Put each of the risk scores into bins
    binned_risks = np.digitize(y_prob, bin_edges) - 1
    binned_risks[binned_risks == num_bins] = num_bins-1
    ece = 0.0
    prob_true_bins = []
    prob_pred_bins = []
    for i in range(num_bins):
        bin_mask = binned_risks == i
        if np.any(bin_mask): # We did have some probability values that landed in this bin
            # Find mean probability of actual trd_positive values in this bin
            mean_prob = np.mean(y_true[bin_mask])
            # Find mean predicted risk score
            mean_score = np.mean(y_prob[bin_mask])
            # Weight of this bin is the number of items in the bin divided by the total number of anchor patients
            bin_weight = np.sum(bin_mask) / y_prob.shape[0]
            # Add to error
            ece += bin_weight * np.abs(mean_prob - mean_score)
            # Add to the bins
            prob_true_bins.append(mean_prob)
            prob_pred_bins.append(mean_score)
   
    # Calibration slope and intercept
    prob_pred_bins = np.array(prob_pred_bins)
    prob_true_bins = np.array(prob_true_bins)
    model = LinearRegression()
    model.fit(prob_pred_bins.reshape(-1,1), prob_true_bins)
    slope, intercept = model.coef_[0], model.intercept_
    
    # Count extreme predictions
    mask = np.zeros_like(y_prob)
    mask[y_prob < 0.1] = 1
    low_proportion = np.sum(mask) / len(mask)
    mask = np.zeros_like(y_prob)
    mask[y_prob > 0.9] = 1
    high_proportion = np.sum(mask) / len(mask)
    
    return {
        'roc_score': roc_score,
        'auprc': auprc,
        'brier_score': brier_score,
        'weighted_calibration_error': ece,
        'calibration_slope': slope,
        'calibration_intercept': intercept,
        'proportion_risk_score_<0.1': low_proportion,
        'proportion_risk_score_>0.9': high_proportion,
    }
    
def run_trd_prediction_computation():
    # Merge all evaluation results from different .csv files into one dataframe
    df = load_neighborhood_data()
            
    anchor_ids = set(df['anchor_patient_id'])
    predictor = TRDPredictor()
    anchor_trd_labels = {
        patient_id: predictor.get_trd_status(candidate_patient_id=patient_id)
        for patient_id in anchor_ids
    }
    
    # Create a report to put in a text file
    text_report = "Prediction Analysis Report\n\n"
    
    # Run the battle
    results = {}
    # Store for each mode and strategy the TRD predictions of each anchor patient
    raw_predictions = [] 
    for scheme, current_df in [(mode, df[df['neighbor_scheme'] == mode.name]) for mode in NeighborScheme]:
        for strat in WeightingStrategy:
            print(f"Running analysis for weighting strategy: {scheme.name}_{strat.name}...", flush=True)
            grouped_by_anchor_patient = current_df.groupby('anchor_patient_id')
            labels = []
            risks = []
            ess_values = []
            for anchor_id, group in grouped_by_anchor_patient:
                risk, ess = calculated_weighted_risk(group=group, strategy=strat)
                labels.append(anchor_trd_labels[anchor_id])
                risks.append(risk)
                ess_values.append(ess)
                raw_predictions.append({
                    'anchor_patient_id': anchor_id,
                    'predicted_risk': risk,
                    'true_label': labels[-1],
                    'ess': ess,
                    'weighting_strategy': strat.name,
                    'neighbor_scheme': scheme.name,
                })
            metrics = compute_metrics(y_true=np.array(labels), y_prob=np.array(risks))
            _, roc_score_ci_low, roc_score_ci_high = plot_receiving_operator_characteristic(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{scheme.name}_{strat.name}')
            plot_precision_recall(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{scheme.name}_{strat.name}')
            plot_calibration(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{scheme.name}_{strat.name}')
            plot_decision_curve_analysis(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{scheme.name}_{strat.name}')
            plot_effective_sample_size_distribution(ess_values=np.array(ess_values), mode=f'{scheme.name}_{strat.name}')
            plot_optimal_confusion_matrix(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{scheme.name}_{strat.name}')
            # Add to text report
            text_report += f"{scheme.name}_{strat.name} Metrics:\n\
    'roc_score': {metrics['roc_score']}\n\
    'roc_score_ci_low': {roc_score_ci_low}\n\
    'roc_score_ci_high': {roc_score_ci_high}\n\
    'auprc': {metrics['auprc']}\n\
    'brier_score': {metrics['brier_score']}\n\
    'weighted_calibration_error': {metrics['weighted_calibration_error']}\n\
    'mean_ESS': {np.mean(np.array(ess_values))}\n\n"
            results[f"{scheme.name}_{strat.name}"] = {
                'roc_score': metrics['roc_score'],
                'roc_score_ci_low': roc_score_ci_low,
                'roc_score_ci_high': roc_score_ci_high,
                'auprc': metrics['auprc'],
                'brier_score': metrics['brier_score'],
                'weighted_calibration_error': metrics['weighted_calibration_error'],
                'Mean_ESS': np.mean(np.array(ess_values))
            }
    
    results_txt_file = Path(os.environ['RESULTS_DIR']) / f'results.txt'
    with open(results_txt_file, 'w') as f:
        f.write(text_report)
    
    # Turn results into a pandas data frame and save the .csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(os.environ['RESULTS_DIR']) / f'summary.csv')
    pd.DataFrame(raw_predictions).to_csv(Path(os.environ['RESULTS_DIR']) / f'summary_predictions.csv')
    print(f"Prediction analysis complete!", flush=True)