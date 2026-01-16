import numpy as np
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.scorer import Scorer

class TRDPredictor:
    
    def __init__(self):
        self.retriever = Retriever()
        self.scorer = Scorer()
        self.alpha = float(os.environ['WEIGHTING_EXPONENT'])
        
        self.trd_set = set()
        trd_file = Path(os.environ['TRD_LIST_PATH'])
        self.trd_set = set(trd_file.read_text().splitlines())
        
    def get_trd_status(self, candidate_id: str) -> int:
        """
        Return 1 if the patient is in the list of TRD positive patients, and 0 otherwise
        
        :param candidate_id: ID of the candidate patient
        :type candidate_id: str
        :return: Integer flag for if the patient is TRD
        :rtype: int
        """
        return 1 if candidate_id in self.trd_set else 0
    
    def predict_risk(self, index_id: str) -> Dict:
        """
        Docstring for predict_risk
        
        :param index_id: Hashed ID of the narrative of the patient of interest
        :type index_id: str
        :return: Information on TRD risk probability along with predictor information
        :rtype: Dict
        """
        patient_id = self.retriever.get_patient_id(id=index_id)
        index_narrative = self.retriever.get_narrative(id=index_id)
        index_vector = self.retriever.get_vector(id=index_id)
        # NOTE - be sure to exclude this patient from the list of viable neighbors
        nearest_neighbors = self.retriever.search(query_vector=index_vector, exclude_id=index_id)
        weights = np.zeros(shape=(len(nearest_neighbors),), dtype=np.float32) # Weights associated with each neighbor
        trds = np.zeros_like(a=weights) # TRD 0/1 flag associated with each neighbor
        neighbor_patient_ids = [] 
        neighbor_scores = np.zeros_like(weights)
        for i, neighbor in enumerate(nearest_neighbors):
            neighbor_narrative_string_id = neighbor[0]
            neighbor_patient_id = self.retriever.get_patient_id(id=neighbor_narrative_string_id)
            neighbor_patient_ids.append(neighbor_patient_id)
            neighbor_narrative = self.retriever.get_narrative(id=neighbor_narrative_string_id)
            score = self.scorer.judge(index_narrative=index_narrative, candidate_narrative=neighbor_narrative, index_id=index_id, candidate_id=neighbor_narrative_string_id)['overall_similarity']
            neighbor_scores[i] = score
            weights[i] = np.power(score/100, self.alpha)
            trds[i] = self.get_trd_status(candidate_id=neighbor_patient_id)
        
        # Calculate predicted TRD probability
        trd_prob = np.dot(a=weights, b=trds.T) / np.sum(a=weights)
        
        # Calculate effective sample size
        weights_squared = np.multiply(weights, weights)
        ess = np.power(np.sum(weights), 2) / np.sum(weights_squared)
        
        # Record risk score
        risk = 'low' if trd_prob < 0.2 else ('moderate' if trd_prob < 0.5 else 'high')
        
        # Record highest 5 contributing neighbor patients
        top_5_indices = np.argpartition(neighbor_scores, -5)[-5:]
        top_5_scores = neighbor_scores[top_5_indices]
        neighbor_patient_ids = np.array(neighbor_patient_ids)
        top_5_ids = neighbor_patient_ids[top_5_indices]
        
        # Return all such information
        return {
            'risk_score' : trd_prob,
            'risk_tier' : risk,
            'confidence_ess' : ess,
            'evidence' : [
                {
                    'neighbor_patient_id' : patient_id,
                    'score' : score
                }
                for patient_id, score in zip(top_5_ids, top_5_scores)
            ]
        }