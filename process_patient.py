from evaluate import evaluate_prediction_by_category
from query_and_response import force_valid_prediction, generate_prompt
from config import get_global_config

def process_patient(patient: dict) -> tuple[str, dict]:
    try:
        prompt = generate_prompt(patient)
        predicted_next_visit = force_valid_prediction(prompt)
        actual_next_visit = patient["visits"][get_global_config().num_visits-1]
        actual = {k: set(actual_next_visit[k]) for k in ["diagnoses", "medications", "treatments"]}
        scores = evaluate_prediction_by_category(predicted_next_visit, actual)
        result = {
                "visit_idx": get_global_config().num_visits-1,
                "predicted": predicted_next_visit,
                "actual": actual,
                "scores": scores
            }
        return patient["patient_id"], result
    except Exception as e:
        print(f"Error processing patient {patient.get('patient_id', 'unknown')}: {e}")
        return patient.get("patient_id", "unknown"), {"error": str(e)}