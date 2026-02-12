import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.shared.utils import load_neighborhood_data

from dotenv import load_dotenv
load_dotenv()

def run_chonology_check():
    """Helper function to evaluate TRD prediction performance over varying chronological lengths of patient history
    """
    # Load the actual prediction risk scores
    risk_scores_df = pd.read_csv(Path(os.environ['RESULTS_DIR']) / 'battle_1_predictions.csv')
    
    # Load patient chronological lengths
    retriever = Retriever()
    lengths_df = pd.DataFrame({
        'id': retriever.ids,
        'chronological_length': retriever.chronological_lengths
    })
    # Merge that into the risk scores dataframe
    risk_scores_df = risk_scores_df.merge(lengths_df, left_on='anchor_id', right_on='id', how='left')
    
    risk_scores_df['prediction_error'] = (risk_scores_df['predicted_risk'] - risk_scores_df['true_label']).abs()
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
        plt.title(f'Error vs. History Length: {weighting_strat}')
        plt.savefig(Path(os.environ['RESULTS_DIR']) / f'battle_1_chronology_check_{weighting_strat}.png')
        plt.close()
        check_results.append({
            'weighting_strategy': weighting_strat,
            'spearman_rho_correlation': correlation,
            'p_value': p_value,
        })
    pd.DataFrame(check_results).to_csv(Path(os.environ['RESULTS_DIR']) / f'battle_1_chronology_check.csv')

def run_cosine_check():
    """Helper function to produce a graph of cosine similarity over random patient pairs versus neighbor patient pairs
    """
    # Load anchor patient neighborhood data frame
    df = load_neighborhood_data()
    
    # Compute anchor to anchor similarities
    retriever = Retriever()
    unique_anchor_ids = df['anchor_id'].unique()
    anchor_indices = np.array([retriever.ids_to_index[id] for id in unique_anchor_ids])
    anchor_vectors = retriever.vectors[anchor_indices]
    sim_matrix = np.dot(anchor_vectors, anchor_vectors.T) # (N x k) x (k x N) -> (N x N) similarities
    unique_pair_sims = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)] # Exclude self pairs
    
    # Plot the two histograms
    plt.figure(figsize=(10,6))
    plt.hist(unique_pair_sims, alpha=0.5, color='red', label='Random Cosine Similarities')
    plt.hist(df['cosine_sim'], alpha=0.5, color='green', label='Neighbor Cosine Similarities')
    plt.legend()
    plt.title('Random vs. Neighborhood Cosine Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(Path(os.environ['RESULTS_DIR']) / 'cosine_score_random_vs_neighbor.png')
    plt.close()