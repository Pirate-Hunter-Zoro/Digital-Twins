import numpy as np
import os
from pathlib import Path
from typing import Dict, List
import random

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.scorer import Scorer

class TRDPredictor:
    
    def __init__(self):
        self.retriever = Retriever()
        self.scorer = Scorer()
        
        self.k_pool = int(os.environ['NUM_NEIGHBOR_PATIENTS'])
        self.k_score = int(os.environ['TRD_TEST_COUNT'])
        
        self.trd_set = set()
        trd_file = Path(os.environ['TRD_LIST_PATH'])
        self.trd_set = set([l.strip('"') for l in trd_file.read_text().splitlines()])
        
    def get_trd_status(self, candidate_id: str) -> int:
        """
        Return 1 if the patient is in the list of TRD positive patients, and 0 otherwise
        
        :param candidate_id: ID of the candidate patient
        :type candidate_id: str
        :return: Integer flag for if the patient is TRD
        :rtype: int
        """
        return 1 if candidate_id in self.trd_set else 0
    
    def construct_neighborhood_data(self, index_id: str) -> list[dict]:
        """Return the information of all the neighbors of this anchor patient

        Args:
            index_id (str): Narrative hash ID of the patient

        Returns:
            list[dict]: Similarity scores, etc. of all neighbor patients
        """
        index_narrative, index_vector, index_patient_id = self.retriever.get_narrative(id=index_id), self.retriever.get_vector(id=index_id), self.retriever.get_patient_id(id=index_id)
        neighbors = self.retriever.search(query_vector=index_vector, exclude_id=index_id)
        for hash_id, _ in neighbors:
            # Quick check to ensure this patient is not included in the neighbors
            if hash_id == index_id:
                raise ValueError(f"ERROR: narrative with hash ID {index_id} was one of its own neighbors...")
            neighbor_patient_id = self.retriever.get_patient_id(id=hash_id)
            if neighbor_patient_id == index_patient_id:
                raise ValueError(f"ERROR: patient with ID {index_patient_id} was one of their own neighbors...")
        # Now sort by cosine similarity
        neighbors.sort(key=lambda x: x[1])
        neighbors = neighbors[:self.k_pool]
        for hash_id, score in neighbors:
            pass