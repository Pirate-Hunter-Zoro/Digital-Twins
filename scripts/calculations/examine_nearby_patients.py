import os
import sys
import json
import pickle
import argparse
from collections import defaultdict

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
# TODO - ENTRAPTA STOP HALLUCINATING
from scripts.config import setup_config, get_global_config
from scripts.eval.scoring_utils import compute_weighted_cosine_score
from scripts.eval.load_neighbors import load_neighbors_for_config
from scripts.eval.load_patient_results import load_all_patient_results
from scripts.eval.idf_utils import load_idf_registry
from scripts.read_data.load_patient_data import load_patient_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", type=str, required=True)
    parser.add_argument("--vectorizer_method", type=str, required=True)
    parser.add_argument("--distance_metric", type=str, default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    args = parser.parse_args()

    # Setup config dynamically
    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )

    config = get_global_config()

    # Load required data
    neighbors_by_patient = load_neighbors_for_config(config)
    predictions = load_all_patient_results(config)
    idf_registry = load_idf_registry()
    all_patients = {p["patient_id"]: p for p in load_patient_data()}

    # Load similarity matrix
    sim_path = os.path.join("data", "term_similarity_matrix.pkl")
    with open(sim_path, "rb") as f:
        similarity_matrix = pickle.load(f)

    # Output paths
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    log_path = "logs/neighbor_analysis_errors.txt"
    output_path = "data/neighbor_analysis_summary.json"

    log_lines = []
    output_data = []

    print("--- Examining neighbors ---")
    for patient_id, neighbors in neighbors_by_patient.items():
        patient = all_patients.get(patient_id)
        predicted = predictions.get(patient_id)

        if not patient or not predicted:
            log_lines.append(f"[SKIP] Missing data for patient {patient_id}")
            continue

        actual = patient["visits"][config.num_visits - 1]

        try:
            scores = compute_weighted_cosine_score(predicted, actual, idf_registry, similarity_matrix)
            output_data.append({
                "patient_id": patient_id,
                "top_neighbors": [n[0][0] for n in neighbors[:3]],
                "prediction_scores": scores
            })
        except Exception as e:
            log_lines.append(f"[ERROR] Failed scoring for patient {patient_id}: {str(e)}")

    # Save results
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"✅ Done. {len(output_data)} patients processed.")
    print(f"📄 Output: {output_path}")
    print(f"🪵 Log: {log_path}")


if __name__ == "__main__":
    main()
