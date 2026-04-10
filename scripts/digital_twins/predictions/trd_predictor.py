import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.scorer import Scorer
from scripts.digital_twins.neighbors.neighbor_scheme import NeighborScheme
from scripts.shared.utils import VectorSource

class TRDPredictor:
    
    def __init__(self, exclude_ids: set[str]=set(), source: VectorSource=VectorSource.EMBEDDING):
        """Create necessary retrievers and scorers and TRD-positive set

        Args:
            exclude_ids (set[str]): Set of anchor patients which are not allowed to be considered neighbors
            source (VectorSource): Determines whether the vectors to use for the prediction scheme are deterministic or the model embeddings
        """
        self.retriever = Retriever(exclude_ids=exclude_ids, source=source)
        self.scorer = Scorer()
        self.trd_set = set()
        trd_file = Path(os.environ['TRD_LIST_PATH'])
        self.trd_set = set([l.strip('"') for l in trd_file.read_text().splitlines()])
        
    def get_trd_status(self, candidate_patient_id: str) -> int:
        """
        Return 1 if the patient is in the list of TRD positive patients, and 0 otherwise
        
        :param candidate_id: ID of the candidate patient
        :type candidate_id: str
        :return: Integer flag for if the patient is TRD
        :rtype: int
        """
        return 1 if candidate_patient_id in self.trd_set else 0
    
    def construct_neighborhood_data(self, index_id: str, scheme: NeighborScheme) -> list[dict]:
        """Return the information of all the neighbors of this anchor patient retrieved according to the given scheme

        Args:
            index_id (str): ID of the patient
            scheme (PredictionScheme): Scheme for how we generate the neighborhood of patients (random, nearest, farthest, etc.)

        Returns:
            list[dict]: Similarity scores, etc. of all neighbor patients
        """
        index_narrative, index_vector, chronological_length =\
            self.retriever.get_narrative(id=index_id),\
                self.retriever.get_vector(id=index_id),\
                        self.retriever.get_chronological_length(id=index_id)
        neighbors = self.retriever.search(query_vector=index_vector, scheme=scheme)
        for neighbor_id, _ in neighbors:
            # Quick check to ensure this patient is not included in the neighbors
            if neighbor_id == index_id:
                raise ValueError(f"ERROR: Patient with ID {index_id} was one of its own neighbors...")
        
        neighborhood_data = []
        for idx, (neighbor_id, score) in enumerate(neighbors):
            neighbor_narrative = self.retriever.get_narrative(id=neighbor_id)
            llm_sim = self.scorer.judge(index_narrative=index_narrative, 
                                        candidate_narrative=neighbor_narrative, 
                                        index_id=index_id, 
                                        candidate_id=neighbor_id)['overall_similarity']
            # Flag for if this neighbor is trd
            neighbor_trd_flag = self.get_trd_status(candidate_patient_id=neighbor_id)
            neighborhood_data.append({
                "neighbor_scheme": scheme.name,
                "chronological_length": chronological_length,
                "anchor_patient_id": index_id,
                "neighbor_patient_id": neighbor_id,
                "cosine_sim": score,
                "llm_sim": llm_sim,
                "neighbor_trd_label": neighbor_trd_flag,
                "rank_cosine": idx+1,
            })
            
        return neighborhood_data