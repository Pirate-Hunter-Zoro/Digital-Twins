import sqlite3
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
import sys

from dotenv import load_dotenv
load_dotenv()

from scripts.models.vllm_client import VllmClient
from scripts.shared.prompts import PromptLoader
from scripts.digital_twins.neighbors.retriever import Retriever

class Scorer:
    
    def __init__(self, require_client:bool = True):
        """Initialize vllm client and internal database

        Args:
            require_client (bool, optional): Boolean for whether or not to load the VLLM server. Defaults to True.
        """
        if require_client:
            self.client = VllmClient()
        self.prompt_loader = PromptLoader()
        self.retriever = Retriever()
        self._init_db()
        
    def _init_db(self):
        vectors_path = Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"
        os.makedirs(vectors_path.parent, exist_ok=True)
        self.connection = sqlite3.connect(vectors_path)
        self.cursor = self.connection.cursor()
        # Create table for judgements
        self.cursor.execute('''
CREATE TABLE IF NOT EXISTS llm_judgements (
    id_a TEXT,
    id_b TEXT,
    overall_score INTEGER,
    full_response TEXT,
    PRIMARY KEY (id_a, id_b)
);
''')
        
    def _get_cached_judgement(self, id_a: str, id_b: str) -> Optional[Dict]:
        """
        Return cached judgement of the pairs if it exists
        
        :param id_a: String id of the first patient
        :type id_a: str
        :param id_b: String id of the second patient
        :type id_b: str
        :return: Resulting judgement if it is present (else None)
        :rtype: Dict | None
        """
        pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        self.cursor.execute('''
SELECT full_response FROM llm_judgements WHERE id_a=? AND id_b=?
''', pair)
        row = self.cursor.fetchone()
        if row != None:
            return json.loads(row[0])
        else:
            return None
    
    def _cache_judge(self, id_a: str, id_b: str, response_json: Dict):
        """
        Extracts LLM judgement score from json and and saves everything to the database

        :param id_a: String id of first patient
        :type id_a: str
        :param id_b: String id of second patient
        :type id_b: str
        :param response_json: Response from the LLM
        :type response_json: Dict
        """
        pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        score = response_json['overall_similarity']
        self.connection.execute('''
INSERT OR REPLACE INTO llm_judgements (id_a, id_b, overall_score, full_response) VALUES (?, ?, ?, ?)
''',
                (pair[0], pair[1], score, json.dumps(response_json, indent=4))
        )
        self.connection.commit()
        
    def judge(self, index_narrative: str, candidate_narrative: str, index_id: str, candidate_id: str) -> Dict[str, Any]:
        """
        Query the LLM judge (if necessary) to retrieve a similarity scoring between the two patients
        
        :param index_narrative: Narrative of patient of interest
        :type index_narrative: str
        :param candidate_narrative: Narrative of neighbor patient
        :type candidate_narrative: str
        :param index_id: String id of the patient of interest
        :type index_id: str
        :param candidate_id: String id of the neighbor patient
        :type candidate_id: str
        :return: LLM response parsed as a dictionary
        :rtype: Dict[str, Any]
        """
        cached = self._get_cached_judgement(id_a=index_id, id_b=candidate_id)
        if cached != None:
            return cached
        # Otherwise we need to ask the LLM for a judgement
        system_prompt = self.prompt_loader.get_judge_system()
        user_prompt = self.prompt_loader.render_judge_user(narrative_a=index_narrative, narrative_b=candidate_narrative)
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        
        # Try to get a judgement
        temperatures = [0.0, 0.4, 0.8]
        for temperature in temperatures: # Multiple attempts at valid response allowed with varying temperatures
            try:
                response = self.client.chat(messages=messages, temperature=temperature)
                cleaned_response = response.strip()
                if "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1].split("```")[0]
                elif "```" in cleaned_response:
                    cleaned_response = cleaned_response.split("```")[1].split("```")[0]
                
                response_json = json.loads(cleaned_response)
                self._cache_judge(id_a=index_id, id_b=candidate_id, response_json=response_json)
                return response_json
            except json.JSONDecodeError as e:
                try:
                    bracket_idx = cleaned_response.rfind(']')
                    if bracket_idx == -1:
                        continue # Try again
                    else:
                        # Add curly brackets to close the json, and 
                        sliced_response = cleaned_response[:bracket_idx+1] + "}"
                        response_json = json.loads(sliced_response)
                        sys.stderr.write(f"Model Struggled and Json had to be truncated when judging patient IDs: {index_id} vs {candidate_id}\n")
                        sys.stderr.write(f"Continuing...\n")
                        sys.stderr.write(f"===============================\n")
                        sys.stderr.flush() # Force the output
                        return response_json
                except:
                    sys.stderr.write(f"IDs: {index_id} vs {candidate_id}\n")
                    sys.stderr.write(f"Response Tail: {cleaned_response[-500:]}\n")
                    sys.stderr.write(f"===============================\n")
                    sys.stderr.flush() # Force the output
                    continue # Try again
            except Exception:
                continue # Try again
        sys.stderr.write(f"COMPLETE REPEATED FAILURE when judging patient IDs: {index_id} vs {candidate_id}\n")
        sys.stderr.write(f"Removing {candidate_id} from {index_id} neighbors...\n")
        sys.stderr.write(f"===============================\n")
        sys.stderr.flush() # Force the output
        return {}