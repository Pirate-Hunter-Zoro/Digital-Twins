import os
import numpy as np
import pandas as pd
from pathlib import Path
from enum import Enum

from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss, 
    precision_recall_curve, 
    auc,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.shared.plots import (
    plot_receiving_operator_characteristic,
    plot_precision_recall,
    plot_calibration,
    plot_decision_curve_analysis,
    plot_effective_sample_size_distribution,
    plot_optimal_confusion_matrix
)
from scripts.shared.utils import load_neighborhood_data

class WeightingStrategy(Enum):
    UNIFORM = "UNIFORM"
    COSINE = "COSINE"
    LLM = "LLM"
    COMBINED = "COMBINED"

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
        raise ValueError(f"ERROR: all weights zero for patient with ID: {patient_id} using strategy {strategy.value}...")
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
    
    # Break patients up into bins and calculate the true and predicted mean TRD-positive probabilities for each patient bin
    prob_true, prob_pred = calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=10)
    # We must weight the bins by their count
    bin_weights = np.histogram(y_prob, bins=10, range=(0,1))[0] / len(y_prob)
    bin_weights = bin_weights[bin_weights != 0] # Calibration curve bins were already non-zero filtered
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    
    # Calibration slope and intercept
    model = LinearRegression()
    model.fit(prob_pred.reshape(-1,1), prob_true)
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
    df_non_random = df[df['is_random_baseline'] == False]
    df_random = df[df['is_random_baseline'] == True]
    anchor_ids = set(df['anchor_patient_id'])
    predictor = TRDPredictor()
    retriever = Retriever()
    anchor_trd_labels = {
        patient_id: predictor.get_trd_status(candidate_patient_id=patient_id)
        for patient_id in anchor_ids
    }
    
    # Slice data frame to only have the nearest K_SCORE cosine similarities
    df_battle_non_random = df_non_random[df_non_random['rank_cosine'] <= int(os.environ['K_SCORE'])]
    df_battle_random = df_random[df_random['rank_cosine'] <= int(os.environ['K_SCORE'])]
    
    # Create a report to put in a text file
    text_report = "Battle 1 Analysis Report\n\n"
    
    # Run the battle
    results = {}
    # Store for each mode and strategy the TRD predictions of each anchor patient
    raw_predictions = [] 
    for mode_name, current_df in [("Non-Random", df_battle_non_random), ("Random", df_battle_random)]:
        weighting_strats = [WeightingStrategy.UNIFORM, WeightingStrategy.COSINE, WeightingStrategy.LLM, WeightingStrategy.COMBINED]
        for strat in weighting_strats:
            print(f"Running analysis for weighting strategy: {mode_name}_{strat.value}...", flush=True)
            grouped_by_anchor_patient = current_df.groupby('anchor_id')
            labels = []
            risks = []
            ess_values = []
            for anchor_hash, group in grouped_by_anchor_patient:
                risk, ess = calculated_weighted_risk(group=group, strategy=strat)
                labels.append(anchor_trd_labels[retriever.get_patient_id(anchor_hash)])
                risks.append(risk)
                ess_values.append(ess)
                raw_predictions.append({
                    'anchor_id': anchor_hash,
                    'anchor_patient_id': retriever.get_patient_id(anchor_hash),
                    'predicted_risk': risk,
                    'true_label': labels[-1],
                    'ess': ess,
                    'strategy': f"{mode_name}_{strat.name}"
                })
            metrics = compute_metrics(y_true=np.array(labels), y_prob=np.array(risks))
            plot_receiving_operator_characteristic(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{mode_name}_{strat.value}')
            plot_precision_recall(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{mode_name}_{strat.value}')
            plot_calibration(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{mode_name}_{strat.value}')
            plot_decision_curve_analysis(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{mode_name}_{strat.value}')
            plot_effective_sample_size_distribution(ess_values=np.array(ess_values), mode=f'{mode_name}_{strat.value}')
            plot_optimal_confusion_matrix(y_true=np.array(labels), y_prob=np.array(risks), mode=f'{mode_name}_{strat.value}')
            # Add to text report
            text_report += f"{mode_name}_{strat.value} Metrics:\n\
    'roc_score': {metrics['roc_score']}\n\
    'auprc': {metrics['auprc']}\n\
    'brier_score': {metrics['brier_score']}\n\
    'weighted_calibration_error': {metrics['weighted_calibration_error']}\n\
    'mean_ESS': {np.mean(np.array(ess_values))}\n\n"
            results[f"{mode_name}_{strat.value}"] = {
                'roc_score': metrics['roc_score'],
                'auprc': metrics['auprc'],
                'brier_score': metrics['brier_score'],
                'weighted_calibration_error': metrics['weighted_calibration_error'],
                'Mean_ESS': np.mean(np.array(ess_values))
            }
    
    results_txt_file = Path(os.environ['RESULTS_DIR']) / 'results.txt'
    with open(results_txt_file, 'w') as f:
        f.write(text_report)
    
    # Turn results into a pandas data frame and save the .csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(os.environ['RESULTS_DIR']) / 'summary.csv')
    pd.DataFrame(raw_predictions).to_csv(Path(os.environ['RESULTS_DIR']) / 'predictions.csv')
    print("Battle 1 analysis complete!", flush=True)