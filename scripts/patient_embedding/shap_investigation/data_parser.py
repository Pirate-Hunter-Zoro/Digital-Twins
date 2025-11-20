from pathlib import Path
from typing import Dict
import os
from dotenv import load_dotenv
from scripts.patient_embedding.shared.narrative_parsing import parse_single_narrative_sections
import random

load_dotenv()

narratives_dir = Path(os.environ['NARRATIVES_DIR'])
num_narratives = int(os.environ['NUM_SHAP_NARRATIVES'])
record_every = 10

def parse_test_narratives() -> Dict[str, Dict[str, str]]:
    # For each patient, return a dictionary of their summary, medications, diagnoses, and full_text
    os.makedirs(narratives_dir, exist_ok=True)
    all_narratives = list(narratives_dir.glob("*.md"))
    random.seed(int(os.environ['SEED']))
    narratives = random.sample(all_narratives, num_narratives)
    results = {}
    for i, narrative in enumerate(narratives):
        id = narrative.stem
        with open(narrative, 'r') as f:
            results[id] = parse_single_narrative_sections(f.read())
        
        if (i+1) % record_every == 0:
            print(f"Parsed {i+1} patient narratives...")
            
    return results