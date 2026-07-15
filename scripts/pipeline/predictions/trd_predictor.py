import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.pipeline.neighbors.retriever import Retriever
from scripts.pipeline.neighbors.scorer import Scorer
from scripts.pipeline.neighbors.neighbor_scheme import NeighborScheme
from scripts.shared.utils import load_trd_set

class TRDPredictor:
    
    def __init__(self, exclude_ids: set[str]=set(), save_time_hist:bool = True):
        """Create necessary retrievers and scorers and TRD-positive set

        Args:
            exclude_ids (set[str]): Set of anchor patients which are not allowed to be considered neighbors
            save_time_hist (bool, optional): Boolean for whether or not to create the time histogram. Defaults to True.
        """
        self.retriever = Retriever(exclude_ids=exclude_ids, save_time_hist=save_time_hist)
        self.judge_sims = int(os.environ['COMPUTE_LLM_SIMILARITY'])==1 
        self.scorer = Scorer(require_client=self.judge_sims, save_time_hist=save_time_hist)
        self.trd_set = set()
        self.trd_set = load_trd_set()
        
    def get_trd_status(self, candidate_patient_id: str) -> int:
        """
        Return 1 if the patient is in the list of TRD positive patients, and 0 otherwise
        
        :param candidate_id: ID of the candidate patient
        :type candidate_id: str
        :return: Integer flag for if the patient is TRD
        :rtype: int
        """
        return 1 if candidate_patient_id in self.trd_set else 0