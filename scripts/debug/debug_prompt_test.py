import sys
import os
import json
from pathlib import Path

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
from scripts.read_data.load_patient_data import load_patient_data
from scripts.calculations.process_patient import (
    generate_prompt,
    force_valid_prediction
)

# --- Main ---
def main():
    print("🧪 Debugging prompt generation + LLM prediction pipeline...")

    patients = load_patient_data()

    # Choose the first valid patient with enough visits
    from scripts.config import setup_config
    setup_config(
        representation_method="bag_of_codes",
        vectorizer_method="biobert-mnli-mednli",
        distance_metric="euclidean",
        num_visits=6,
        num_patients=5000,
        num_neighbors=5,
    )

    for patient in patients:
        if len(patient.get("visits", [])) >= 6:
            print(f"🎯 Using Patient ID: {patient['patient_id']}")
            break
    else:
        raise ValueError("No patients with at least 6 visits!")

    prompt = generate_prompt(patient)
    print("\n📝 Generated Prompt:\n" + "-" * 60)
    print(prompt)
    print("-" * 60)

    prediction = force_valid_prediction(prompt)
    print("\n🔮 LLM Prediction:\n" + "-" * 60)
    print(json.dumps(prediction, indent=2))
    print("-" * 60)


if __name__ == "__main__":
    main()
