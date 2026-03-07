import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from scipy.stats import spearmanr

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import random

RESULTS_DIR = Path(os.environ['RESULTS_DIR'])
os.makedirs(RESULTS_DIR, exist_ok=True)

from scripts.shared.utils import load_neighborhood_data

def vector_analysis(vectors: np.array) -> np.array:
    """Analyzes shape and norm of all the given vectors

    Args:
        vectors (np.array): All vectors to perform analysis on

    Returns:
        np.array: Norm of all vectors
    """
    ndim = vectors.shape
    with open(RESULTS_DIR / 'vectors_shape.txt', 'w') as f:
        f.write(f"Vector Dimensions Over All Patients: {ndim}")
    
    # Compute norm of all vectors
    vector_norms = np.linalg.norm(vectors, axis=1)
    min_norm = np.min(vector_norms)
    max_norm = np.max(vector_norms)
    plt.figure(figsize=(10,6))
    if np.isclose(min_norm, max_norm):
        plt.hist(vector_norms, range=(min_norm-0.01, max_norm+0.01), bins=1)
    else:
        plt.hist(vector_norms)
    plt.title("Histogram of Norms of Vectorized Patients")
    plt.xlabel("Vector Norm")
    plt.ylabel("Frequency")
    plt.savefig(str(RESULTS_DIR) / 'vector_norms.png')
    plt.close()
    return vector_norms
    
def cone_analysis(vectors: np.array, vector_norms: np.array) -> tuple[list[tuple[int,int]], np.array]:
    """Analyze baseline random cosine similarity value

    Args:
        vectors (np.array): All vectors of interest
        vector_norms (np.array): Norms of all vectors of interest

    Returns:
        tuple[list[tuple[int,int]], np.array]: Resulting random pairs generated and their cosine similarities
    """
    # Using itertools will be too large when generating all possible combinations - just make unique pairs until we have enough
    num_pairs = 5000
    pairs = set()
    indices = range(vectors.shape[0])
    while len(pairs) < num_pairs:
        new_pair = random.sample(indices, 2)
        new_pair = (new_pair[0], new_pair[1]) if new_pair[0] < new_pair[1] else (new_pair[1], new_pair[0])
        pairs.add(new_pair)
    pairs = list(pairs)
    a_indices = [pair[0] for pair in pairs]
    b_indices = [pair[1] for pair in pairs]
    
    # Now grab the vectors and perform analysis - if any vectors are of zero magnitute this will crash and burn as it SHOULD because that should never happen
    a_vectors = vectors[a_indices]
    a_vectors = a_vectors / vector_norms[a_indices][:, np.newaxis] # (N,) -> (N,1)
    b_vectors = vectors[b_indices]
    b_vectors = b_vectors / vector_norms[b_indices][:, np.newaxis]
    random_similarities = np.sum(a_vectors * b_vectors, axis=1)
    # We want to compare this with the cosine similarities from our results
    results_df = load_neighborhood_data()
    neighbor_cosines = results_df['cosine_sim']
    
    # Create a histogram comparing the random similarities with the neighbor similarities
    plt.figure(figsize=(10,6))
    plt.hist(random_similarities, label="Random Similarity Values", alpha=0.5, density=True, bins=100)
    plt.hist(neighbor_cosines, label="Neighbor Cosine Similarities", alpha=0.5, density=True, bins=100)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(str(RESULTS_DIR / 'cos_random_vs_neighbor.png'))
    plt.close()
    
    return (pairs, random_similarities)
    
def monotonicity_analysis(vectors: np.array, random_pairs: list[tuple[int,int]], cosine_sims: np.array):
    """Compare Euclidean and Cosine similarity metrics of random pairs

    Args:
        vectors (np.array): Vectors of interest
        random_pairs (list[tuple[int,int]]): Pre-computed random pairs
        cosine_sims (np.array): Cosine similarities of all pairs pre-computed
    """
    a_vectors = vectors[[a_idx for a_idx, _ in random_pairs]]
    b_vectors = vectors[[b_idx for _, b_idx in random_pairs]]
    euclidean_distances = np.linalg.norm(a_vectors-b_vectors, axis=1)
    cosine_distances = 1 - cosine_sims
    rho, p_value = spearmanr(euclidean_distances, cosine_distances)
    plt.figure(figsize=(10,6))
    plt.scatter(euclidean_distances, cosine_distances, alpha=0.1, s=1)
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Cosine Distances")
    plt.title(f"Euclidean vs. Cosine (Rho={rho}, P-Value={p_value})")
    plt.savefig(str(RESULTS_DIR / 'cos_vs_euclidean.png'))
    plt.close()

def main():
    vectors_db_path = Path(os.environ['VECTORS_DIR']) / 'vectors.db'
    connection = sqlite3.connect(vectors_db_path)
    cursor = connection.cursor()
    # Grab all the vectors and perform some analysis
    cursor.execute('''
SELECT vector FROM vectors
''')
    vectors = np.array([np.frombuffer(vec[0], dtype=np.float32) for vec in cursor.fetchall()])
    
    # Perform analyses
    vector_norms = vector_analysis(vectors=vectors)
    random_pairs, cosine_sims = cone_analysis(vectors=vectors, vector_norms=vector_norms)
    monotonicity_analysis(vectors=vectors, random_pairs=random_pairs, cosine_sims=cosine_sims)
    
    connection.close()
    
if __name__=="__main__":
    main()