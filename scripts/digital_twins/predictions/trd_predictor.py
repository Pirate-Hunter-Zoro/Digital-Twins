import numpy as np
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.scorer import Scorer
from scripts.models.patient_embedder import PatientEmbedder

class TRDPredictor:
    
    def __init__(self):
        self.retriever = Retriever()
        self.scorer = Scorer()
        self.embedder = PatientEmbedder()
        self.alpha = float(os.environ['WEIGHTING_EXPONENT'])
        
        self.trd_set = set()
        trd_file = Path(os.environ['TRD_LIST_PATH'])
        with open(trd_file, 'r') as f:
            for line in trd_file.read_text().split("\n"):
                self.trd_set.add(line.strip())
        
    def _get_trd_status(self, candidate_id: str) -> int:
        """
        Return 1 if the patient is in the list of TRD positive patients, and 0 otherwise
        
        :param candidate_id: ID of the candidate patient
        :type candidate_id: str
        :return: Integer flag for if the patient is TRD
        :rtype: int
        """
        return 1 if candidate_id in self.trd_set else 0
    
    def predict_risk(self, index_narrative: str, index_id: str) -> Dict:
        """
        Docstring for predict_risk
        
        :param index_narrative: Narrative of patient of interest
        :type index_narrative: str
        :param index_id: ID 
        :type index_id: str
        :return: Information on TRD risk probability along with predictor information
        :rtype: Dict
        """
        patient_id = self.retriever.get_patient_id(id=index_id)
        index_vector = self.embedder.vectorize(([patient_id], [index_narrative]))[0]
        nearest_neighbors = self.retriever.search(query_vector=index_vector)
        for neighbor in nearest_neighbors:
            pass
        