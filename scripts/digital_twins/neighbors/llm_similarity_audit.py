from pathlib import Path
import os
import json
import sqlite3

from dotenv import load_dotenv
load_dotenv()

from scripts.digital_twins.neighbors.scorer import Scorer

def main():
    # Use a scorer's connection to the judgements database
    scorer = Scorer(require_client=False)
    
    # Create a connection to the vectors database
    vector_cursor = sqlite3.connect(Path(os.environ['VECTORS_DIR']) / "vectors.db").cursor()
    
    # Find the lowest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()
    for row in rows:
        # Create report including the first narrative, the second narrative, and the third narrative
        id_a, id_b, response = row
        # Fetch two corresponding narratives
        
    
    # Find the highest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()