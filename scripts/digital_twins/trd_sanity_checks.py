import os
from itertools import combinations
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.shared.similarity import cosine

from dotenv import load_dotenv
load_dotenv()

def run_chonology_check():
    """Helper function to evaluate TRD prediction performance over varying chronological lengths of patient history
    """
    # Load the actual prediction risk scores
    retriever = Retriever()
    risk_scores_df = pd.read_csv(Path(os.environ['RESULTS_DIR']) / 'battle_1_predictions.csv')
    risk_scores_df['chronological_length'] = [retriever.get_time_length(id) for id in risk_scores_df['anchor_id']]
    risk_scores_df['prediction_error'] = [abs(predicted_risk - true_label) for predicted_risk, true_label in zip(risk_scores_df['predicted_risk'], risk_scores_df['true_label'])]
    grouped_risk_scores_df = risk_scores_df.groupby('strategy') # Group results by weighting strategy used
    check_results = []
    for weighting_strat, results in grouped_risk_scores_df:
        chronological_lengths = results['chronological_length']
        prediction_error = results['prediction_error']
        # Find correlation between time length and prediction error
        correlation, p_value = spearmanr(np.array(chronological_lengths), np.array(prediction_error))
        # Create scatter plot
        plt.figure(figsize=(10,6))
        plt.scatter(chronological_lengths, prediction_error)
        plt.xlabel('Chronological Length (Days) of Patient History')
        plt.ylabel('TRD Probability Prediction Error')
        plt.title('TRD Prediction Error vs. Chronological Length')
        plt.savefig(Path(os.environ['RESULTS_DIR']) / f'battle_1_chronology_check_{weighting_strat.name}.png')
        plt.close()
        check_results.append({
            'weighting_strategy': weighting_strat.name,
            'spearman_rho_correlation': correlation,
            'p_value': p_value,
        })
    pd.DataFrame(check_results).to_csv(Path(os.environ['RESULTS_DIR']) / f'battle_1_chronology_check_{weighting_strat.name}.csv')

def run_cosine_check():
    """Helper function to produce a graph of cosine similarity over random patient pairs versus neighbor patient pairs
    """
    # Load neighbor similarities
    df = pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob('trd_evaluation_results_*.csv')])
    anchor_to_neighbor_cos_sims = df['cosine_sim']
    anchor_patient_ids = df['anchor_id'] # Narrative hash IDs of each anchor patient
    anchor_to_anchor_cos_sims = np.array([cosine(id_a, id_b) for (id_a, id_b) in combinations(anchor_patient_ids.tolist(), 2)])
    
    # Plot the two histograms
    plt.figure(figsize=(10,6))
    plt.hist(anchor_to_anchor_cos_sims, alpha=0.5, color='red', label='Random Cosine Similarities')
    plt.hist(anchor_to_neighbor_cos_sims, alpha=0.5, color='green', label='Neighbor Cosine Similarities')
    plt.legend()
    plt.title('Random vs. Neighborhood Cosine Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(Path(os.environ['RESULTS_DIR']) / 'cosine_score_random_vs_neighbor.png')
    plt.close()