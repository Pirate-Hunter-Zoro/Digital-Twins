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

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.digital_twins.neighbors.retriever import Retriever
# TODO - plotting and .txt report

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
        weights = np.array([max(0,score) for score in cleaned_group['cosine_sim']])
    elif strategy == WeightingStrategy.COMBINED:
        weights = np.array([max(cos_score,0)*(llm_score/100)**alpha  for llm_score, cos_score in zip(cleaned_group['llm_sim'], cleaned_group['cosine_sim'])])
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
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_prob)
    auprc = auc(x=recall, y=precision)
    # Brier score
    brier_score = brier_score_loss(y_true=y_true, y_prob=y_prob)
    # Break patients up into bins and calculate the true and predicted mean TRD-positive probabilities for each patient bin
    prob_true, prob_pred = calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    return {
        'roc_score': roc_score,
        'auprc': auprc,
        'brier_score': brier_score,
        'expected_calibration_error': ece
    }
    
def run_analysis():
    # Merge all evaluation results from different .csv files into one dataframe
    df = pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob("trd_evaluation_results_*.csv")], ignore_index=True)
    anchor_ids = set(df['anchor_patient_id'])
    predictor = TRDPredictor()
    retriever = Retriever()
    anchor_labels = {
        patient_id: predictor.get_trd_status(candidate_id=patient_id)
        for patient_id in anchor_ids
    }
    
    # Slice data frame to only have cosine ranks of less than or equal to K_SCORE
    df_battle = df[df['rank_cosine'] <= int(os.environ['K_SCORE'])]
    
    # Run the battle
    weighting_strats = [WeightingStrategy.UNIFORM, WeightingStrategy.COSINE, WeightingStrategy.LLM, WeightingStrategy.COMBINED]
    results = {}
    for strat in weighting_strats:
        grouped_by_anchor_patient = df_battle.groupby('anchor_id')
        labels = []
        risks = []
        ess_values = []
        for anchor_hash, group in grouped_by_anchor_patient:
            risk, ess = calculated_weighted_risk(group=group, strategy=strat)
            labels.append(anchor_labels[retriever.get_patient_id(anchor_hash)])
            risks.append(risk)
            ess_values.append(ess)
        metrics = compute_metrics(y_true=np.array(labels), y_prob=np.array(risks))
        results[strat.value] = {
            'roc_score': metrics['roc_score'],
            'auprc': metrics['auprc'],
            'brier_score': metrics['brier_score'],
            'expected_calibration_error': metrics['expected_calibration_error'],
            'Mean_ESS': np.mean(np.array(ess_values))
        }
    
    # Turn results into a pandas data frame and save the .csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(Path(os.environ['RESULTS_DIR']) / 'battle_1_summary.csv')