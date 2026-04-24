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
    embedding_cursor = sqlite3.connect(Path(os.environ['EMBEDDINGS_DIR']) / "embeddings.db").cursor()
    results_dir = Path(os.environ['RESULTS_DIR']) / 'llm_audit/'
    os.makedirs(results_dir, exist_ok=True)
    
    # Find the lowest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()
    for row in rows:
        # Create report including the first narrative, the second narrative, and the third narrative
        patient_id_a, patient_id_b, response = row
        # Fetch two corresponding narratives
        embedding_cursor.execute('''
SELECT text FROM embeddings WHERE patient_id=?
''', (patient_id_a,))
        narrative_a = embedding_cursor.fetchone()[0]
        embedding_cursor.execute('''
SELECT text FROM embeddings WHERE patient_id=?
''', (patient_id_b,))
        narrative_b = embedding_cursor.fetchone()[0]
        response_indented = json.dumps(json.loads(response), indent=4)
        
        # Write the two narratives and the response to a .txt file named after the two patients
        with open(results_dir / f"judgement_{patient_id_a}_{patient_id_b}.txt", 'w') as f:
            f.write(f"Narrative A:\n{narrative_a}\n\n\n\
                Narrative B:\n{narrative_b}\n\n\n\
                    Response:\n{response_indented}")
        
    
    # Find the highest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score DESC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()
    rows = scorer.cursor.fetchall()
    for row in rows:
        # Create report including the first narrative, the second narrative, and the third narrative
        patient_id_a, patient_id_b, response = row
        # Fetch two corresponding narratives
        embedding_cursor.execute('''
SELECT text FROM embeddings WHERE patient_id=?
''', (patient_id_a,))
        narrative_a = embedding_cursor.fetchone()[0]
        embedding_cursor.execute('''
SELECT text FROM embeddings WHERE patient_id=?
''', (patient_id_b,))
        narrative_b = embedding_cursor.fetchone()[0]
        response_indented = json.dumps(json.loads(response), indent=4)
        
        # Write the two narratives and the response to a .txt file named after the two patients
        with open(results_dir / f"judgement_{patient_id_a}_{patient_id_b}.txt", 'w') as f:
            f.write(f"Narrative A:\n{narrative_a}\n\n\n\
Narrative B:\n{narrative_b}\n\n\n\
Response:\n{response_indented}")
    
if __name__=="__main__":
    main()