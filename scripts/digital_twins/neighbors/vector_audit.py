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

def vector_analysis(vectors: np.array):
    """Analyzes shape and norm of all the given vectors

    Args:
        vectors (np.array): All vectors to perform analysis on
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
    
def cone_analysis(vectors: np.array):
    """Analyze baseline random cosine similarity value

    Args:
        vectors (np.array): All vectors of interest
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
        np.dot(vectors[a_idx], vectors[b_idx]) / (np.linalg.norm(vectors[a_idx])*np.linalg.norm(vectors[b_idx]))
        for a_idx, b_idx in pairs
    ])
    # We want to compare this with the cosine similarities from our results
    results_df = load_trd_prediction_csv_results()
    neighbor_cosines = results_df['cosine_sim']
    
    # Create a histogram comparing the random similarities with the neighbor similarities

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
    vector_analysis(vectors=vectors)
    cone_analysis(vectors=vectors)
    
    connection.close()