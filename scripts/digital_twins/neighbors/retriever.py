import sqlite3
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

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
SELECT id, vector, chronological_length FROM vectors
            """
        )
        # List of tuples - (id string of narrative, vector in bytes)
        self.patient_vectors = self.cursor.fetchall()
        
        self.ids = []
        vectors = []
        self.chronological_lengths = []
        for row in self.patient_vectors:
            self.ids.append(row[0])
            vectors.append(np.frombuffer(row[1], dtype=np.float32))
            self.chronological_lengths.append(row[2])
        self.ids_to_index = {id: i for i, id in enumerate(self.ids)}
        self.vectors = np.vstack(vectors)
        # Normalize each vector
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors /= norms
        
        # Create histogram of chronological lengths
        plt.figure(figsize=(10,6))
        plt.hist(np.array(self.chronological_lengths), bins=100)
        plt.xlabel("Chronological Length (Days)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Patient Chronological History Lengths")
        plt.savefig(Path(os.environ['RESULTS_DIR']) / 'history_length_histogram.png')
        plt.close()
        
    def get_narrative(self, id: str) -> Optional[str]:
        """
        Helper method to return the narrative corresponding with the input hashed ID calculated from the narrative
        
        :param id: Hashed ID from narrative
        :type id: str
        :return: Narrative producing inputted hash ID
        :rtype: str
        """
        self.cursor.execute(
            """
SELECT text FROM vectors WHERE id=?
            """,
            (id,))
        row = self.cursor.fetchone()
        if row is not None:
            return row[0]
        else:
            return None
    
    def get_patient_id(self, id: str) -> Optional[str]:
        """
        Helper method to return the patient id corresponding with the narrative that has the input hashed ID calculated from the narrative
        
        :param id: Hashed ID from narrative of patient
        :type id: str
        :return: Respective patient ID
        :rtype: str
        """
        self.cursor.execute(
            """
SELECT patient_id FROM vectors WHERE id=?
            """,
            (id,))
        row = self.cursor.fetchone()
        if row is not None:
            return row[0]
        else:
            return None

    def get_vector(self, id: str) -> np.array:
        """
        Helper method to return the vector corresponding with the narrative that has the input hashed ID calculated from the narrative
        
        :param id: Hashed ID from narrative of patient
        :type id: str
        :return: Respective vector
        :rtype: np.array
        """
        self.cursor.execute(
            """
SELECT vector FROM vectors WHERE id=?
            """,
            (id,))
        row = self.cursor.fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32)
        else:
            return None
        
    def get_time_length(self, id: str) -> int:
        """
        Helper method to return the chronological length in days of the corresponding with the narrative that has the input hashed ID calculated from the narrative
        
        :param id: Hashed ID from narrative of patient
        :type id: str
        :return: Respective vector
        :rtype: np.array
        """
        self.cursor.execute(
            """
SELECT chronological_length FROM vectors WHERE id=?
            """,
            (id,))
        row = self.cursor.fetchone()
        if row is not None:
            return row[0]
        else:
            return None
        
    def search(self, query_vector: np.array, exclude_id: str=None) -> List[Tuple[str, float]]:
        """
        Find the nearest patients to this patient in terms of cosine similarity of the vectors
        
        :param query_vector: Vector of interest which came from a patient
        :type query_vector: np.array
        :param exclude_id: ID to exclude when considering nearby neighbors (e.g. don't let the patient themself count as a neighbor)
        :type query_vector: str
        :return: Resulting nearest patients and their similarity scores
        :rtype: List[Tuple[str, float]]
        """
        k = int(os.environ['NUM_NEIGHBOR_PATIENTS'])
        # Normalize query vector
        mag = np.linalg.norm(query_vector)
        if mag > 0:
            normalized_vector = query_vector / mag
        # Find the cosine similarity of this vector with all other vectors in our database
        similarities = self.vectors @ normalized_vector # NOTE - due to normalization, dot product IS cosine similarity
        if exclude_id != None and exclude_id in self.ids_to_index.keys():
            # Kill this similarity score so it won't be one of the top ones
            idx = self.ids_to_index[exclude_id]
            similarities[idx] = -1.0
        
        # Go through and find the kth largest value, and ensure everything to the right is bigger than it, so grab the last k values of this partitioned array to get the largest similarity values
        unsorted_top_k_indices = np.argpartition(similarities, -k)[-k:]
        unsorted_top_k_scores = similarities[unsorted_top_k_indices]
        # Sort only the most similar indices
        sorted_k_indices = np.argsort(unsorted_top_k_scores)[::-1] # DESCENDING sorting order - return the indices that would put the highest similarity scores first
        top_k_indices = unsorted_top_k_indices[sorted_k_indices]
        top_k_scores = unsorted_top_k_scores[sorted_k_indices]
        
        return [(self.ids[index], score) for index, score in zip(top_k_indices, top_k_scores)]