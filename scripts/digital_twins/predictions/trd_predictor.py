import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.retriever import Retriever
from scripts.digital_twins.neighbors.scorer import Scorer

class TRDPredictor:
    
    def __init__(self):
        self.retriever = Retriever()
        self.scorer = Scorer()
        
        self.k_pool = int(os.environ['NUM_NEIGHBOR_PATIENTS'])
        self.k_score = int(os.environ['K_SCORE'])
        
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
    
    def construct_neighborhood_data(self, index_id: str, random: bool=False) -> list[dict]:
        """Return the information of all the neighbors of this anchor patient

        Args:
            index_id (str): Narrative hash ID of the patient
            random (bool, optional): Whether the neighborhood is random or picked by an embedding model. Defaults to False.

        Returns:
            list[dict]: Similarity scores, etc. of all neighbor patients
        """
        index_narrative, index_vector, index_patient_id, chronological_length =\
            self.retriever.get_narrative(id=index_id),\
                self.retriever.get_vector(id=index_id),\
                    self.retriever.get_patient_id(id=index_id),\
                        self.retriever.get_chronological_length(id=index_id)
        neighbors = self.retriever.search(query_vector=index_vector, exclude_id=index_id, random=random)
        for neighbor_narrative_hash_id, _ in neighbors:
            # Quick check to ensure this patient is not included in the neighbors
            if neighbor_narrative_hash_id == index_id:
                raise ValueError(f"ERROR: narrative with hash ID {index_id} was one of its own neighbors...")
            neighbor_patient_id = self.retriever.get_patient_id(id=neighbor_narrative_hash_id)
            if neighbor_patient_id == index_patient_id:
                raise ValueError(f"ERROR: patient with ID {index_patient_id} was one of their own neighbors...")
        
        # Now sort by decreasing cosine similarity and grab information on all neighbors
        neighborhood_data = []
        neighbors.sort(key=lambda x: x[1], reverse=True)
        neighbors = neighbors[:self.k_pool]
        for idx, (neighbor_narrative_hash_id, score) in enumerate(neighbors):
            neighbor_patient_id = self.retriever.get_patient_id(id=neighbor_narrative_hash_id)
            if idx < self.k_score:
                # This is an 'important enough' neighbor by cosine metric
                neighbor_narrative = self.retriever.get_narrative(id=neighbor_narrative_hash_id)
                llm_sim = self.scorer.judge(index_narrative=index_narrative, 
                                            candidate_narrative=neighbor_narrative, 
                                            index_id=index_id, 
                                            candidate_id=neighbor_narrative_hash_id)['overall_similarity']
            else:
                # Filler for the background density check
                llm_sim = None
            # Flag for if this neighbor is trd
            neighbor_trd_flag = self.get_trd_status(candidate_patient_id=neighbor_patient_id)
            neighborhood_data.append({
                "is_random_baseline": random,
                "anchor_id": index_id,
                "chronological_length": chronological_length,
                "anchor_patient_id": index_patient_id,
                "neighbor_id": neighbor_narrative_hash_id,
                "neighbor_patient_id": neighbor_patient_id,
                "cosine_sim": score,
                "llm_sim": llm_sim,
                "neighbor_trd_label": neighbor_trd_flag,
                "rank_cosine": idx+1,
            })
            
        return neighborhood_data