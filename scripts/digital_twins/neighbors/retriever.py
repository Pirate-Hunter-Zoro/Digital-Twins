import sqlite3
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

VECTORS_DIR = Path(os.environ['VECTORS_DIR'])

class Retriever:
    
    def __init__(self):
        """
        Loads all patient vectors and their corresponding narrative string IDs
        """
        vectors_db_path = VECTORS_DIR / "vectors.db"
        self.connection = sqlite3.connect(vectors_db_path)
        self.cursor = self.connection.cursor()
        
        # Load all patient vectors
        self.cursor.execute(
            """
SELECT id, vector FROM vectors
            """
        )
        # List of tuples - (id string of narrative, vector in bytes)
        self.patient_vectors = self.cursor.fetchall()
        self.connection.close()
        
        self.ids = []
        vectors = []
        for row in self.patient_vectors:
            self.ids.append(row[0])
            vectors.append(np.frombuffer(row[1], dtype=np.float32))
        self.vectors = np.vstack(vectors)
        # Normalize each vector
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors /= norms
        
    def search(self, query_vector: np.array) -> List[Tuple[str, float]]:
        """
        Find the nearest patients to this patient in terms of cosine similarity of the vectors
        
        :param query_vector: Vector of interest which came from a patient
        :type query_vector: np.array
        :return: Resulting nearest patients and their similarity scores
        :rtype: List[Tuple[str, float]]
        """
        k = int(os.environ['NUM_NEIGHBOR_PATIENTS'])
        # Normalize query vector
        mag = np.linalg.norm(query_vector)
        if mag > 0:
            query_vector /= np.linalg.norm(query_vector)
        # Find the cosine similarity of this vector with all other vectors in our database
        similarities = self.vectors @ query_vector # NOTE - due to normalization, dot product IS cosine similarity
        # Go through and find the kth largest value, and ensure everything to the right is bigger than it, so grab the last k values of this partitioned array to get the largest similarity values
        unsorted_top_k_indices = np.argpartition(similarities, -k)[-k:]
        unsorted_top_k_scores = similarities[unsorted_top_k_indices]
        # Sort only the most similar indices
        sorted_k_indices = np.argsort(unsorted_top_k_scores)[::-1] # DESCENDING sorting order - return the indices that would put the highest similarity scores first
        top_k_indices = unsorted_top_k_indices[sorted_k_indices]
        top_k_scores = unsorted_top_k_scores[sorted_k_indices]
        
        return [(self.ids[index], score) for index, score in zip(top_k_indices, top_k_scores)]