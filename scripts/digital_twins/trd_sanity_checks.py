import os
import random
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.shared.similarity import cosine

from dotenv import load_dotenv
load_dotenv()

def main():
    # Get a random sample of pairs
    retriever = Retriever()
    pair_ids = set()
    all_hash_ids = retriever.ids
    pairs = []
    for _ in range(int(os.environ['NUM_PAIRS_SANITY_CHECK'])):
        pair_indices = random.sample(range(len(all_hash_ids)), 2)
        pair_indices = (pair_indices[0], pair_indices[1]) if pair_indices[0] < pair_indices[1] else (pair_indices[1], pair_indices[0])
        while pair_indices in pair_ids:
            pair_indices = random.sample(range(len(all_hash_ids)), 2)
            pair_indices = (pair_indices[0], pair_indices[1]) if pair_indices[0] < pair_indices[1] else (pair_indices[1], pair_indices[0])  
        pair_ids.add(pair_indices)
        first_idx = pair_indices[0]
        second_idx = pair_indices[1]
        pairs.append((all_hash_ids[first_idx], all_hash_ids[second_idx]))
    
    # Compute cosine similarities of all pairs
    random_cos_sims = np.array([cosine(id_a=pair[0], id_b=pair[1]) for pair in pairs])
    
    # Load neighbor similarities
    df = pd.concat([pd.read_csv(f) for f in Path(os.environ['RESULTS_DIR']).glob('trd_evaluation_results_*.csv')])
    neighbor_cos_sims = df['cosine_sim']
    
    # Plot the two histograms
    plt.figure(figsize=(10,6))
    plt.hist(random_cos_sims, alpha=0.5, color='red', label='Random Cosine Similarities')
    plt.hist(neighbor_cos_sims, alpha=0.5, color='green', label='Neighbor Cosine Similarities')
    plt.legend()
    plt.title('Random vs. Neighborhood Cosine Similarity Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(Path(os.environ['RESULTS_DIR']) / 'cosine_score_random_vs_neighbor.png')
    plt.close()