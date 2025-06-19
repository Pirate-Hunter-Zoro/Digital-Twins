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

from scripts.config import get_global_config

def load_patient_data() -> list[dict]:
    """
    Loads real patient data from all_patients_combined.json and transforms it
    to match the 'visits' structure expected by downstream scripts.
    """
    global_config = get_global_config()

    # Define the path to your combined real patient data JSON file
    # This path should match output_json_dir from process_data.py
    real_data_combined_json_path = os.path.join(
        project_root, # Use project_root to build path to real_data
        "real_data", # This is the folder name based on your output_json_dir
        "all_patients_combined.json"
    )

    if not os.path.exists(real_data_combined_json_path):
        raise FileNotFoundError(f"Combined patient data not found at: {real_data_combined_json_path}. Please ensure process_data.py has run successfully.")

    print(f"Loading patient data from: {real_data_combined_json_path}")
    with open(real_data_combined_json_path, 'r') as f:
        raw_patients_data = json.load(f)

    processed_patients = []
    for patient in raw_patients_data:
        patient_id = patient["patient_id"]
        demographics = patient["demographics"]

        # Transform 'encounters' into 'visits'
        visits = []
        for encounter in patient.get("encounters", []):
            # Each encounter dictionary becomes a visit dictionary
            # Start with details flattened into the visit object
            visit = encounter.get("details", {})
            
            # Add diagnoses, medications, and map procedures to treatments
            # These are already lists of dicts from process_data.py output
            visit["diagnoses"] = encounter.get("diagnoses", [])
            visit["medications"] = encounter.get("medications", [])
            visit["treatments"] = encounter.get("procedures", []) # Map 'procedures' from real data to 'treatments'

            visits.append(visit)
        
        # Sort visits by 'StartVisit' date for chronological history
        try:
            # pd.to_datetime will convert string dates to datetime objects.
            # errors='coerce' will turn unparseable dates into NaT (Not a Time).
            visits.sort(key=lambda v: pd.to_datetime(v.get("StartVisit"), errors='coerce'))
            
            # Re-assign sequential visit_idx after sorting
            for i, visit in enumerate(visits):
                visit["visit_idx"] = i 
        except Exception as e:
            print(f"Warning: Could not sort visits by StartVisit for patient {patient_id}: {e}. Keeping original order and assigning sequential visit_idx.", file=sys.stderr)
            # If sorting fails, just ensure visit_idx is assigned sequentially based on original order
            for i, visit in enumerate(visits):
                visit["visit_idx"] = i 

        processed_patients.append({
            "patient_id": patient_id,
            "demographics": demographics,
            "visits": visits # This now matches the expected 'visits' structure
        })

    return processed_patients