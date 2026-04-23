import os
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme
from scripts.shared.utils import load_neighborhood_data

from dotenv import load_dotenv
load_dotenv()

def run_chonology_check():
    """
    Helper function to evaluate TRD prediction performance over varying chronological lengths of patient history
    """
    # Load the actual prediction risk scores
    risk_scores_df = pd.read_csv(Path(os.environ['RESULTS_DIR']) / f'summary_predictions.csv')
    
    # Load patient chronological lengths
    retriever = Retriever()
    lengths_df = pd.DataFrame({
        'id': retriever.ids,
        'chronological_length': retriever.chronological_lengths
    })
    # Merge that into the risk scores dataframe
    risk_scores_df = risk_scores_df.merge(lengths_df, left_on='anchor_patient_id', right_on='id', how='left')
    
    risk_scores_df['prediction_error'] = (risk_scores_df['predicted_risk'] - risk_scores_df['true_label']).abs()
    grouped_risk_scores_df = risk_scores_df.groupby(['weighting_strategy', 'neighbor_scheme']) # Group results by weighting strategy used
    check_results = []
    for (weighting_strat, neighbor_scheme), results in grouped_risk_scores_df:
        chronological_lengths = results['chronological_length']
        prediction_error = results['prediction_error']
        # Find correlation between time length and prediction error
        correlation, p_value = spearmanr(np.array(chronological_lengths), np.array(prediction_error))
        # Create scatter plot
        plt.figure(figsize=(10,6))
        plt.scatter(chronological_lengths, prediction_error)
        plt.xlabel('Chronological Length (Days) of Patient History')
        plt.ylabel('TRD Probability Prediction Error')
        plt.title(f'Error vs. History Length: {neighbor_scheme}_{weighting_strat}')
        save_path = Path(os.environ['RESULTS_DIR']) / 'chronology_checks' / f'chronology_check_{neighbor_scheme}_{weighting_strat}.png'
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(str(save_path))
        plt.close()
        check_results.append({
            'weighting_strategy': weighting_strat,
            'spearman_rho_correlation': correlation,
            'p_value': p_value,
        })
    pd.DataFrame(check_results).to_csv(Path(os.environ['RESULTS_DIR']) / f'chronology_check.csv')

def run_cosine_check():
    """Helper function to produce a graph of cosine similarity over random patient pairs versus neighbor patient pairs
    """
    # Load anchor patient neighborhood data frame
    df = load_neighborhood_data()
    df = df[df['neighbor_scheme'] == NeighborScheme.NEAREST.name]
    
    # Compute anchor to anchor similarities
    retriever = Retriever()
    unique_anchor_ids = df['anchor_patient_id'].unique()
    anchor_indices = np.array([retriever.ids_to_index[id] for id in unique_anchor_ids])
    anchor_vectors = retriever.vectors[anchor_indices]
    sim_matrix = np.dot(anchor_vectors, anchor_vectors.T) # (N x k) x (k x N) -> (N x N) similarities
    unique_pair_sims = sim_matrix[np.triu_indices(sim_matrix.shape[0], k=1)] # Exclude self pairs
    
    # Plot the two histograms
    plt.figure(figsize=(10,6))
    plt.hist(unique_pair_sims, alpha=0.5, color='red', label='Random Cosine Similarities', bins=100)
    plt.hist(df['cosine_sim'], alpha=0.5, color='green', label='Neighbor Cosine Similarities', bins=100)
    plt.legend()
    plt.title(f'Random vs. Neighborhood Cosine Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(Path(os.environ['RESULTS_DIR']) / f'cosine_score_random_vs_neighbor.png')
    plt.close()
    
def run_trd_sanity_checks():
    run_chonology_check()
    run_cosine_check()