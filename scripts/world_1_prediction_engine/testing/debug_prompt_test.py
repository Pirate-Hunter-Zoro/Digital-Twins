import sys
import os
import json
import argparse
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.llm.query_and_response import (
    setup_prompt_generation,
    generate_prompt,
    force_valid_prediction
)
from scripts.common.config import setup_config

# --- Utility: Convert sets to JSON-serializable lists ---
def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

# --- CLI argument parsing ---
def parse_debug_args():
    parser = argparse.ArgumentParser(description="Debug Prompt Tester")
    parser.add_argument("--representation_method", type=str, required=True)
    parser.add_argument("--vectorizer_method", type=str, required=True)
    parser.add_argument("--distance_metric", type=str, default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    return parser.parse_args()

# --- MAIN! ---
def main():
    args = parse_debug_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors
    )

    setup_prompt_generation()

    print("🧪 Debugging prompt generation + LLM prediction pipeline...")

    patients = load_patient_data()
    for patient in patients:
        if len(patient.get("visits", [])) >= args.num_visits:
            print(f"🎯 Using Patient ID: {patient['patient_id']}")
            break
    else:
        raise ValueError("No patients with sufficient visits!")

    prompt = generate_prompt(patient)
    print("\n📝 Generated Prompt:\n" + "-" * 60)
    print(prompt)
    print("-" * 60)

    prediction = force_valid_prediction(prompt)
    print("\n🔮 LLM Prediction:\n" + "-" * 60)
    print(json.dumps(convert_sets_to_lists(prediction), indent=2))
    print("-" * 60)

if __name__ == "__main__":
    main()
