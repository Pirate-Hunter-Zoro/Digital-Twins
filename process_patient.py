from evaluate import evaluate_prediction_by_category
from query_llm import clean_response
from query_and_response import force_valid_prediction, generate_prompt, parse_llm_response
import json

stored_results = {}
try:
    with open("debug_prompts_and_responses.jsonl", "r") as f:
        for line in f:
            conversation = json.loads(line)
            stored_results[conversation["prompt"]] = conversation["response"]
except Exception as e:
    print(f"⚠️ Could not load stored results: {e}")

def process_patient(patient: dict) -> tuple[str, list[dict]]:
    try:
        results = []
        for visit_idx in range(1, len(patient["visits"])):
            prompt = generate_prompt(patient, visit_idx)
            if prompt in stored_results:
                predicted_next_visit = parse_llm_response(clean_response(stored_results[prompt]).strip())
            else:
                predicted_next_visit = force_valid_prediction(prompt)
            actual_next_visit = patient["visits"][visit_idx]
            actual = {k: set(actual_next_visit[k]) for k in ["diagnoses", "medications", "treatments"]}
            scores = evaluate_prediction_by_category(predicted_next_visit, actual)
            results.append({
                "visit_idx": visit_idx + 1,
                "predicted": predicted_next_visit,
                "actual": actual,
                "scores": scores
            })
        return patient["patient_id"], results
    except Exception as e:
        print(f"⚠️ Error processing patient {patient.get('patient_id', 'unknown')}: {e}")
        return patient.get("patient_id", "unknown"), {"error": str(e)}