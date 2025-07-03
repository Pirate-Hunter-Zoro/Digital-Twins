import os
import sys
import json
import argparse

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    args = parser.parse_args()

    # Build result path from config-like values
    results_path = os.path.join(
        project_root,
        "data",
        f"results_{args.num_patients}_{args.num_visits}_{args.representation_method}_{args.vectorizer_method}_{args.distance_metric}.json"
    )

    if not os.path.exists(results_path):
        print(f"❌ File not found: {results_path}")
        return

    with open(results_path, "r") as f:
        results = json.load(f)

    all_predicted_terms = set()
    for patient_id, data in results.items():
        for category in ["diagnoses", "medications", "treatments"]:
            all_predicted_terms.update(data["prediction"].get(category, []))

    print(f"🔍 Total unique predicted terms: {len(all_predicted_terms)}")
    missing = [term for term in sorted(all_predicted_terms) if not term.strip()]
    if missing:
        print("⚠️ Empty or missing terms found:")
        for term in missing:
            print(f"- '{term}'")
    else:
        print("✅ All terms appear to be non-empty and valid.")

if __name__ == "__main__":
    main()
