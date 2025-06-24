import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.analyze_results.evaluate import evaluate_prediction_by_category
from scripts.llm.query_and_response import force_valid_prediction, generate_prompt
from scripts.config import get_global_config

def process_patient(patient: dict, idf_registry: dict, embedding_library: dict) -> tuple[str, dict]:
    """
    Processes a single patient's data to generate a prediction for their next visit,
    evaluate it against the actual visit, and return the results.

    Args:
        patient (dict): A dictionary containing the patient's data, including their ID and a list of visits.
        idf_registry (dict): A dictionary mapping medical terms to their IDF scores.
        embedding_library (dict): A dictionary mapping medical terms to their vector embeddings.

    Returns:
        tuple[str, dict]: A tuple containing the patient's ID and a dictionary with either the
                          prediction results or an error message.
    """
    try:
        # Check if the patient has enough visits before generating a prompt
        if len(patient["visits"]) < get_global_config().num_visits:
            return patient["patient_id"], {"error": f"Not enough visits ({len(patient['visits'])}) for prediction with num_visits={get_global_config().num_visits}"}

        # Generate the prompt and get a prediction from the LLM
        prompt = generate_prompt(patient)
        predicted_next_visit = force_valid_prediction(prompt)

        # --- FIX APPLIED ---
        # 1. Define the actual_next_visit using the correct index from the config
        actual_next_visit = patient["visits"][get_global_config().num_visits - 1]

        # 2. Extract the specific string values from the lists of dictionaries
        actual_dx_list = [
            d.get("Diagnosis_Name") for d in actual_next_visit.get("diagnoses", []) if d.get("Diagnosis_Name")
        ]
        actual_med_list = [
            m.get("MedSimpleGenericName") for m in actual_next_visit.get("medications", []) if m.get("MedSimpleGenericName")
        ]
        actual_proc_list = [
            p.get("CPT_Procedure_Description") for p in actual_next_visit.get("treatments", []) if p.get("CPT_Procedure_Description")
        ]

        # 3. Create the 'actual' dictionary from the clean lists of strings, now ready for set conversion
        actual = {
            "diagnoses": set(actual_dx_list),
            "medications": set(actual_med_list),
            "treatments": set(actual_proc_list)
        }
        
        # Evaluate the prediction against the actual data
        scores = evaluate_prediction_by_category(predicted_next_visit, actual, idf_registry, embedding_library)

        # Compile the final result object
        result = {
                "visit_idx": get_global_config().num_visits - 1,
                "predicted": predicted_next_visit,
                "actual": actual,
                "scores": scores
            }
        return patient["patient_id"], result
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error processing patient {patient.get('patient_id', 'unknown')}: {e}", file=sys.stderr)
        return patient.get("patient_id", "unknown"), {"error": str(e)}

