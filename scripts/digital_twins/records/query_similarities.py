import sqlite3
import os
from pathlib import Path
import random
import json

from dotenv import load_dotenv
load_dotenv()

EMBEDDINGS_DB = Path(os.environ['EMBEDDINGS_DIR']) / "embeddings.db"
JUDGEMENTS_DB = Path(os.environ['JUDGEMENTS_DIR']) / "judgements.db"
OUTPUT_DIR = Path("test_data/judgements")
SAMPLE_SIZE = 10

def main():
    embeddings_connection = sqlite3.connect(EMBEDDINGS_DB)
    embeddings_cursor = embeddings_connection.cursor()
    judgements_connection = sqlite3.connect(JUDGEMENTS_DB)
    judgements_cursor = judgements_connection.cursor()
    
    # Grab some judgements
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    judgements_cursor.execute("""
SELECT patient_id_a, patient_id_b, overall_score, full_response FROM llm_judgements                          
""")
    rows = judgements_cursor.fetchall()
    for row in random.sample(rows, min(len(rows), SAMPLE_SIZE)):
        id_a, id_b, overall_score, full_response = row
        # Grab the narratives
        embeddings_cursor.execute("""
SELECT text, patient_id FROM embeddings WHERE patient_id = ?
""", (id_a,))
        narrative_a, patient_id_a = embeddings_cursor.fetchone()
        embeddings_cursor.execute("""
SELECT text, patient_id FROM embeddings WHERE patient_id = ?
""", (id_b,))
        narrative_b, patient_id_b = embeddings_cursor.fetchone()
        
        header = f"COMPARISION: {patient_id_a} vs {patient_id_b}"
        
        narrative_a_section = f"FIRST NARRATIVE:\n{narrative_a}"
        narrative_b_section = f"SECOND NARRATIVE:\n{narrative_b}"
        
        overall_score_section = f"OVERALL SCORE: {overall_score}"
        
        full_response = json.loads(full_response)
        full_response_section = f"FULL RESPONSE: {json.dumps(full_response, indent=4)}"
        
        contents = "\n\n".join([header, narrative_a_section,  narrative_b_section, overall_score_section, full_response_section])
        
        txt_file = OUTPUT_DIR / f"{patient_id_a}_vs_{patient_id_b}.txt"
        with open(txt_file, 'w') as f:
            f.write(contents)
            
    embeddings_connection.close()
    judgements_connection.close()
            
if __name__ == "__main__":
    main()