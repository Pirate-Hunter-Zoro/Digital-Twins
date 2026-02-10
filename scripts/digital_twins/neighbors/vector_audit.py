import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import random

from scripts.shared.utils import load_trd_prediction_csv_results

def vector_analysis(vectors: np.array) -> np.array:
    """Analyzes shape and norm of all the given vectors

    Args:
        vectors (np.array): All vectors to perform analysis on

    Returns:
        np.array: Norm of all vectors
    """
    ndim = vectors.shape
    with open(Path(os.environ['RESULTS_DIR']) / 'vectors_shape.txt', 'w') as f:
        f.write(f"Vector Dimensions Over All Patients: {ndim}")
    
    # Compute norm of all vectors
    vector_norms = np.linalg.norm(vectors, axis=1)
    plt.figure(figsize=(10,6))
    plt.hist(vector_norms)
    plt.title("Histogram of Norms of Vectorized Patients")
    plt.xlabel("Vector Norm")
    plt.ylabel("Frequency")
    plt.savefig(str(Path(os.environ['RESULTS_DIR']) / 'vector_norms.png'))
    plt.close()
    return vector_norms
    
def cone_analysis(vectors: np.array, vector_norms: np.array) -> set[tuple[int,int]]:
    """Analyze baseline random cosine similarity value

    Args:
        vectors (np.array): All vectors of interest
        vector_norms (np.array): Norms of all vectors of interest

    Returns:
        set[tuple[int,int]]: Resulting random pairs generated
    """
    # Using itertools will be too large when generating all possible combinations - just make unique pairs until we have enough
    num_pairs = 5000
    pairs = set()
    indices = range(vectors.shape[0])
    while len(pairs) < num_pairs:
        new_pair = random.sample(indices, 2)
        new_pair = (new_pair[0], new_pair[1]) if new_pair[0] < new_pair[1] else (new_pair[1], new_pair[0])
        pairs.add(new_pair)
    
    # Now grab the vectors and perform analysis - if any vectors are of zero magnitute this will crash and burn as it SHOULD because that should never happen
    random_similarities = np.array([
        np.dot(vectors[a_idx], vectors[b_idx]) / (vector_norms[a_idx] * vector_norms[b_idx])
        for a_idx, b_idx in pairs
    ])
    # We want to compare this with the cosine similarities from our results
    results_df = load_trd_prediction_csv_results()
    neighbor_cosines = results_df['cosine_sim']
    
    # Create a histogram comparing the random similarities with the neighbor similarities
    plt.figure(figsize=(10,6))
    plt.hist(random_similarities, label="Random Similarity Values", alpha=0.5, density=True)
    plt.hist(neighbor_cosines, label="Neighbor Cosine Similarities", alpha=0.5, density=True)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(str(Path(os.environ['RESULTS_DIR']) / 'cos_random_vs_neighbor.png'))
    plt.close()
    
    return pairs
    
def monotonicity_analysis(vectors: np.array, random_pairs: set[tuple[int,int]]):
    """Compare Euclidean and Cosine similarity metrics of random pairs

    Args:
        vectors (np.array): Vectors of interest
        random_pairs (set[tuple[int,int]]): Pre-computed random pairs
    """
    a_vectors = vectors[[a_idx for a_idx, _ in random_pairs]]
    b_vectors = vectors[[b_idx for _, b_idx in random_pairs]]
    euclidean_distances = np.linalg.norm(a_vectors-b_vectors, axis=1)
    # TODO - cosine distance

def main():
    vectors_db_path = Path(os.environ['VECTORS_DIR']) / 'vectors.db'
    connection = sqlite3.connect(vectors_db_path)
    cursor = connection.cursor()
    # Grab all the vectors and perform some analysis
    cursor.execute('''
SELECT vector FROM vectors
''')
    vectors = np.array([np.frombuffer(vec, dtype=np.float32) for vec in cursor.fetchall()])
    
    # Perform analyses
    vector_norms = vector_analysis(vectors=vectors)
    random_pairs = cone_analysis(vectors=vectors, norms=vector_norms)
    monotonicity_analysis(vectors=vectors, random_pairs=random_pairs)
    
    connection.close()