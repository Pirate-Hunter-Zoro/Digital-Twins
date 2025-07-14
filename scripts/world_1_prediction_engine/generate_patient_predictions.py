import sys
import os
import json
import argparse
import pickle
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Imports from our new, refactored scripts! ---
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.config import setup_config, get_global_config
from scripts.common.llm.query_llm import query_llm
# We'll import the prompt generation logic directly from our World 1 script!
from scripts.world_1_prediction_engine.generate_patient_predictions import generate_prediction_prompt 

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

# --- MAIN! ---
def main():
    parser = argparse.ArgumentParser(description="Debug Prompt Tester for World 1")
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", required=True)
    parser.add_argument("--num_visits", type=int, required=True)
    parser.add_argument("--num_patients", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--patient_id_to_test", type=str, default=None, help="Specify a patient ID to test. If None, a random one is chosen.")
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name
    )
    config = get_global_config()

    print("--- 🧪 Debugging World 1 Prompt Generation 🧪 ---")

    # --- Data Loading ---
    print("📂 Loading patient and neighbor data...")
    patient_data = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patient_data}

    # Construct the path to the neighbors file from World 2
    neighbors_path = project_root / "data" / config.representation_method / config.vectorizer_method / f"visits_{config.num_visits}" / f"patients_{config.num_patients}" / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}" / "neighbors.pkl"
    
    neighbors_data = None
    if os.path.exists(neighbors_path):
        print(f"✅ Found and loaded neighbors file from: {neighbors_path}")
        with open(neighbors_path, "rb") as f:
            neighbors_data = pickle.load(f)
    else:
        print(f"⚠️ Neighbors file not found at {neighbors_path}. Proceeding without neighbor data for this test.")

    # --- Select a Patient to Test ---
    if args.patient_id_to_test:
        test_patient = patient_lookup.get(args.patient_id_to_test)
        if not test_patient or len(test_patient.get("visits", [])) <= config.num_visits:
            print(f"Error: Patient {args.patient_id_to_test} not found or has insufficient visits. Choosing a random patient.")
            test_patient = None
    else:
        test_patient = None

    if not test_patient:
        # Find a random patient with enough visits
        for patient in patient_data:
            if len(patient.get("visits", [])) > config.num_visits:
                test_patient = patient
                break
    
    if not test_patient:
        raise ValueError("No patients with sufficient visits found in the dataset!")

    print(f"🎯 Using Patient ID: {test_patient['patient_id']} for the test.")

    # --- Generate and Print the Prompt ---
    prompt = generate_prediction_prompt(test_patient, patient_lookup, neighbors_data)
    print("\n📝 Generated Prompt:\n" + "-" * 60)
    print(prompt)
    print("-" * 60)

    # --- Query the LLM and Print the Prediction ---
    print("\nQuerying LLM...")
    raw_response = query_llm(prompt, max_tokens=1024)
    
    prediction = {}
    try:
        # A simple way to extract JSON from a markdown block
        json_match = raw_response.split("```json")[1].split("```")[0]
        prediction = json.loads(json_match)
    except (IndexError, json.JSONDecodeError) as e:
        print(f"⚠️ Could not parse prediction from LLM response. Error: {e}")
        prediction = {"raw_response": raw_response}

    print("\n🔮 LLM Prediction:\n" + "-" * 60)
    print(json.dumps(convert_sets_to_lists(prediction), indent=2))
    print("-" * 60)

if __name__ == "__main__":
    main()