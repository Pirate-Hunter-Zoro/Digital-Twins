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
        vector_cursor.execute('''
SELECT text FROM vectors WHERE id=?
''', (id_a,))
        narrative_a = vector_cursor.fetchone()
        vector_cursor.execute('''
SELECT text FROM vectors WHERE id=?
''', (id_b,))
        narrative_b = vector_cursor.fetchone()
        response_indented = json.dumps(json.loads(response), indent=4)
        # TODO - ensure everything in this 'row in rows' loop is correct and then write to txt file named cleverly
    
    # Find the highest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()