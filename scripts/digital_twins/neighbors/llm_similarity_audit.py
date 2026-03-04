from pathlib import Path
import os
import json

from scripts.digital_twins.neighbors.scorer import Scorer

def main():
    scorer = Scorer(require_client=False)
    # Use the scorer's connection to the database
    
    # Find the lowest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()
    
    
    # Find the highest 5 overall scores
    scorer.cursor.execute('''
SELECT id_a, id_b, full_response FROM llm_judgements ORDER BY overall_score ASC LIMIT 5;
''')
    rows = scorer.cursor.fetchall()