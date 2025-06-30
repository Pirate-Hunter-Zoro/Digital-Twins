import json
from pathlib import Path
from scripts.config import get_global_config

def process_patient_records(raw_data):
    processed = {}
    for patient_id, entry in raw_data.items():
        processed[patient_id] = {
            "visit_idx": entry.get("visit_idx", -1),
            "predicted": entry.get("predicted", {}),
            "actual": entry.get("actual", {})
        }
    return processed

def main():
    config = get_global_config()
    ROOT_DIR = Path(__file__).resolve().parents[2]
    
    input_path = ROOT_DIR / "data" / "raw_predictions.json"  # adjust if needed
    output_filename = f"patient_results_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.json"
    output_path = ROOT_DIR / "data" / output_filename

    print(f"📂 Loading raw data from: {input_path}")
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    print("🔍 Processing patient records...")
    cleaned_data = process_patient_records(raw_data)

    print(f"💾 Saving cleaned patient predictions (NO SCORES) to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)

    print("✅ Done! Your patient records are clean and evaluation-free.")

if __name__ == "__main__":
    main()
