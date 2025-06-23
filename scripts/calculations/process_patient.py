import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

from scripts.analyze_results.evaluate import evaluate_prediction_by_category
from scripts.llm.query_and_response import force_valid_prediction, generate_prompt
from scripts.config import get_global_config

def process_patient(patient: dict, idf_registry: dict, embedding_library: dict) -> tuple[str, dict]:
    try:
        prompt = generate_prompt(patient) 
        predicted_next_visit = force_valid_prediction(prompt)

        if len(patient["visits"]) < get_global_config().num_visits:
            return patient["patient_id"], {"error": f"Not enough visits ({len(patient['visits'])}) for prediction with num_visits={get_global_config().num_visits}"}

        actual_next_visit = patient["visits"][get_global_config().num_visits - 1]

        actual = {
            "diagnoses": set(actual_next_visit.get("diagnoses", [])),
            "medications": set(actual_next_visit.get("medications", [])),
            "treatments": set(actual_next_visit.get("treatments", []))
        }
        
        scores = evaluate_prediction_by_category(predicted_next_visit, actual, idf_registry, embedding_library)

        result = {
                "visit_idx": get_global_config().num_visits - 1,
                "predicted": predicted_next_visit,
                "actual": actual,
                "scores": scores
            }
        return patient["patient_id"], result
    except Exception as e:
        print(f"Error processing patient {patient.get('patient_id', 'unknown')}: {e}", file=sys.stderr)
        return patient.get("patient_id", "unknown"), {"error": str(e)}