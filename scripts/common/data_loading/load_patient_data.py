import os
import sys
import json
import pandas as pd

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

from scripts.common.config import get_global_config

import os
import sys
import json
import pandas as pd
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import get_global_config

def load_patient_data() -> list[dict]:
    """
    Loads real patient data from all_patients_combined.json and transforms it
    to match the 'visits' structure expected by downstream scripts.
    """
    
    # --- FIX APPLIED: Corrected the path to the data directory ---
    data_combined_json_path = project_root / "data" / "all_patients_combined.json"

    if not os.path.exists(data_combined_json_path):
        raise FileNotFoundError(f"Combined patient data not found at: {data_combined_json_path}. Please ensure process_data.py has run successfully.")

    print(f"Loading patient data from: {data_combined_json_path}")
    with open(data_combined_json_path, 'r') as f:
        raw_patients_data = json.load(f)

    processed_patients = []
    for patient in raw_patients_data:
        patient_id = patient["patient_id"]
        demographics = patient["demographics"]
        
        visits = []
        for encounter in patient.get("encounters", []):
            visit = encounter.get("details", {})
            visit["diagnoses"] = encounter.get("diagnoses", [])
            visit["medications"] = encounter.get("medications", [])
            visit["treatments"] = encounter.get("procedures", [])
            visits.append(visit)
        
        try:
            visits.sort(key=lambda v: pd.to_datetime(v.get("StartVisit"), errors='coerce'))
            for i, visit in enumerate(visits):
                visit["visit_idx"] = i 
        except Exception as e:
            print(f"Warning: Could not sort visits by StartVisit for patient {patient_id}: {e}. Keeping original order.", file=sys.stderr)
            for i, visit in enumerate(visits):
                visit["visit_idx"] = i 

        processed_patients.append({
            "patient_id": patient_id,
            "demographics": demographics,
            "visits": visits
        })

    return processed_patients