import os
from itertools import combinations
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.predictions.trd_predictor import TRDPredictor
from scripts.shared.similarity import cosine

from dotenv import load_dotenv
load_dotenv()

def run_chonology_check():
    """Helper function to evaluate TRD prediction performance over varying chronological lengths of patient history
    """
    results_df = pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob('trd_evaluation_results_*.csv')])
    retriever = Retriever()
    predictor = TRDPredictor()
    unique_ids = set(results_df['anchor_id'].items).tolist()
    lengths_df = pd.DataFrame({
        'id': unique_ids,
        'chronological_length': [retriever.get_time_length(id) for id in unique_ids],
    })
    merged_results_df = results_df.merge(lengths_df, on='id')
    merged_results_df['trd_status'] = [predictor.get_trd_status(patient_id) for patient_id in merged_results_df['anchor_patient_id'].items]
    # TODO - error scores

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