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
except:
    pass

def process_patient(patient: dict) -> tuple[str, list[dict]]:
    results = []
    for visit_idx in range(1,len(patient["visits"])):
        # Predict the NEXT visit based on the visits up to visit_idx - only query the llm if we have not already gotten a response from it
        prompt = generate_prompt(patient, visit_idx)
        if prompt in stored_results.keys():
            predicted_next_visit = parse_llm_response(clean_response(stored_results[prompt]).strip())
        else:
            predicted_next_visit = force_valid_prediction(prompt)
        actual_next_visit = patient["visits"][visit_idx] # The NEXT visit given our current visit_idx
        actual = {k: set(actual_next_visit[k]) for k in ["diagnoses", "medications", "treatments"]}
        scores = evaluate_prediction_by_category(
            predicted_next_visit, actual
        )
        result = {
            "visit_idx": visit_idx + 1,
            "predicted": predicted_next_visit,
            "actual": actual,
            "scores": scores
        }
        results.append(result)
    return (patient["patient_id"], results)