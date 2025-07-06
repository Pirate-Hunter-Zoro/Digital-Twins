import os
import sys
import argparse
import json

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.config import setup_config, get_global_config
from scripts.llm.query_and_response import setup_prompt_generation, generate_prediction_output
from scripts.read_data.load_patient_data import load_patient_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", required=True)
    parser.add_argument("--num_visits", type=int, required=True)
    parser.add_argument("--num_patients", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
    )

    config = get_global_config()

    setup_prompt_generation()
    patients = load_patient_data()

    results = {}
    for patient in patients:
        try:
            prediction = generate_prediction_output(patient)
            results[patient["patient_id"]] = prediction
        except Exception as e:
            print(f"Skipping patient {patient['patient_id']}: {e}")
            continue

    output_file = os.path.join(
        project_root,
        "data",
        f"patient_results_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved predictions to: {output_file}")

if __name__ == "__main__":
    main()
