import sqlite3
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme

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
SELECT patient_id, vector, chronological_length FROM vectors
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
        plt.hist(np.array([l for l in self.chronological_lengths if l < int(os.environ['HISTORY_LENGTH_CUTOFF'])]), bins=100)
        plt.xlabel("Chronological Length (Days)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Patient Chronological History Lengths")
        plt.savefig(Path(os.environ['RESULTS_DIR']) / 'history_length_histogram.png')
        plt.close()
        
    def get_narrative(self, id: str) -> str:
        """
        Helper method to return the narrative corresponding with the input ID of the patient
        
        :param id: patient ID
        :type id: str
        :return: Narrative of patient
        :rtype: str
        """
        self.cursor.execute(
            """
SELECT text FROM vectors WHERE patient_id=?
            """,
            (id,))
        return self.cursor.fetchone()[0]
        
    def get_vector(self, id: str) -> np.array:
        """
        Helper method to return the vector corresponding with the narrative that has the input patient ID
        
        :param id: Patient ID of patient with corresponding narrative
        :type id: str
        :return: Respective vector
        :rtype: np.array
        """
        self.cursor.execute(
            """
SELECT vector FROM vectors WHERE patient_id=?
            """,
            (id,))
        return np.frombuffer(self.cursor.fetchone()[0], dtype=np.float32)
        
    def get_chronological_length(self, id: str) -> int:
        """
        Helper method to return the chronological length in days of the corresponding with the narrative of the patient with the corresponding ID
        
        :param id: ID of patient
        :type id: str
        :return: Respective vector of patient's narrative
        :rtype: np.array
        """
        self.cursor.execute(
            """
SELECT chronological_length FROM vectors WHERE patient_id=?
            """,
            (id,))
        return self.cursor.fetchone()[0]
        
    def search(self, query_vector: np.array, exclude_id: str, scheme: NeighborScheme) -> List[Tuple[str, float]]:
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
        
        if scheme.name == NeighborScheme.NEAREST: # Find nearest neighbors by cosine similarity
            sorted_indices = np.argsort(similarities)[::-1] # Highest similarity first
            top_k_indices = sorted_indices[1:k+1] # The very nearest will be the id we need to exclude - assert this is the case or someone is calling this function wrong or the embeddings are wack
            if not self.ids[sorted_indices[0]] == exclude_id:
                raise ValueError(f"Error - ID to exclude should be the patient ID of {self.ids[sorted_indices[0]]}, but was {exclude_id}...")
            top_k_scores = similarities[top_k_indices]
            return [(self.ids[index], score) for index, score in zip(top_k_indices, top_k_scores)]
        elif scheme.name == NeighborScheme.FARTHEST:
            sorted_indices = np.argsort(similarities)[::-1] # Lowest similarity first
            top_k_indices = sorted_indices[:k]
            top_k_scores = similarities[top_k_indices]
            return [(self.ids[index], score) for index, score in zip(top_k_indices, top_k_scores)]
        elif scheme.name == NeighborScheme.SUBSAMPLE:
            # Subsample method
            sample_size = int(os.environ['SUBSAMPLE_POOL_SIZE'])
            available_ids = self.ids.copy()
            available_ids.remove(exclude_id)
            available_ids = np.array(available_ids)
            random_neighbors = np.random.choice(available_ids, size=sample_size, replace=False).tolist()
            corresponding_similarities = similarities[[self.ids_to_index[id] for id in random_neighbors]]
            nearest_similarities_indices_from_sample = np.argsort(corresponding_similarities)[::-1][:k]
            nearest_ids_from_sample = random_neighbors[nearest_similarities_indices_from_sample]
            return [(id, similarities[self.ids_to_index[id]]) for id in nearest_ids_from_sample]
        else:
            # Random neighbors
            available_ids = self.ids.copy()
            available_ids.remove(exclude_id)
            available_ids = np.array(available_ids)
            random_neighbors = np.random.choice(available_ids, size=k, replace=False).tolist()
            
            return [(id, similarities[self.ids_to_index[id]]) for id in random_neighbors]