import sqlite3
import json
import os
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from scripts.models.vllm_client import VllmClient
from scripts.shared.prompts import PromptLoader

class Scorer:
    
    def __init__(self):
        """
        Initialize vllm client and internal database
        """
        self.client = VllmClient()
        self.prompt_loader = PromptLoader()
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
        :type id_b: Dict
        """
        pair = (id_a, id_b) if id_a <= id_b else (id_b, id_a)
        score = response_json['overall_similarity']
        self.connection.executemany(
                '''
INSERT OR REPLACE INTO llm_judgements (id, vector, text, length) VALUES (?, ?, ?, ?)
''',
                (pair[0], pair[1], score, json.dumps(response_json))
        )
        self.connection.commit()