import sys
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from sentence_transformers import SentenceTransformer


# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent  # /.../scripts/analyze_results
scripts_dir = current_script_dir.parent               # /.../scripts
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
# --- End of dynamic sys.path adjustment ---

from config import setup_config, get_global_config
from common.utils import turn_to_sentence
from read_data.load_patient_data import load_patient_data

project_root = scripts_dir.parent  # /.../project_root
data_dir = os.path.join(project_root, "data")

def get_visit_histories(patient_visits: list[dict]) -> dict[int, str]:
    visit_histories = {}
    config = get_global_config()
    history_window_length = config.num_visits
    representation_method = config.representation_method

    if len(patient_visits) < history_window_length:
        return visit_histories

    start_idx = len(patient_visits) - history_window_length
    relevant_visits = patient_visits[start_idx:]
    end_idx = len(patient_visits) - 1

    if representation_method == "visit_sentence":
        history_string = " | ".join(turn_to_sentence(visit) for visit in relevant_visits)
    elif representation_method == "bag_of_codes":
        all_codes = []
        for visit in relevant_visits:
            all_codes.extend([d.get("Diagnosis_Name") for d in visit.get("diagnoses", []) if d.get("Diagnosis_Name")])
            all_codes.extend([m.get("MedSimpleGenericName") for m in visit.get("medications", []) if m.get("MedSimpleGenericName")])
            all_codes.extend([p.get("CPT_Procedure_Description") for p in visit.get("treatments", []) if p.get("CPT_Procedure_Description")])
        history_string = " ".join(filter(None, all_codes))
    else:
        raise ValueError(f"Unknown representation method: {representation_method}")

    visit_histories[end_idx] = history_string
    return visit_histories

def get_visit_vectors(patient_data: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    all_visit_strings = []
    all_keys = []

    for patient in patient_data:
        patient_id = patient["patient_id"]
        visit_histories = get_visit_histories(patient["visits"])
        for end_idx, visit_string in visit_histories.items():
            all_keys.append((patient_id, end_idx))
            all_visit_strings.append(visit_string)

    vectorizer_path = os.path.join(project_root, "models", get_global_config().vectorizer_method)
    vectorizer_instance = SentenceTransformer(vectorizer_path, local_files_only=True)

    print(f"Encoding {len(all_visit_strings)} visit strings...")
    vectors = vectorizer_instance.encode(
        all_visit_strings,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return {key: vec for key, vec in zip(all_keys, vectors)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors
    )
    config = get_global_config()

    # === Check for existing output ===
    neighbors_path = os.path.join(
        data_dir,
        f"neighbors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.pkl"
    )
    if os.path.exists(neighbors_path):
        print(f"✅ Neighbors already computed at {neighbors_path}. Skipping.")
        return

    print("🔍 Computing new nearest neighbors...")
    patient_data = load_patient_data()
    vectors_dict = get_visit_vectors(patient_data)

    all_vectors_path = os.path.join(
        data_dir,
        f"all_vectors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.model_name}.pkl"
    )
    with open(all_vectors_path, "wb") as f:
        pickle.dump(vectors_dict, f)
    print(f"💾 Saved vector embeddings to {all_vectors_path}")

    keys = list(vectors_dict.keys())
    matrix = np.vstack([vectors_dict[k] for k in keys])

    if config.distance_metric == "euclidean":
        sim_matrix = -euclidean_distances(matrix)
    elif config.distance_metric == "cosine":
        sim_matrix = cosine_similarity(matrix)
    else:
        raise ValueError(f"Unsupported distance metric: {config.distance_metric}")

    neighbors = {}
    for i, key in enumerate(keys):
        sims = sim_matrix[i]
        ranked = sorted(
            [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if j != i],
            key=lambda x: x[1],
            reverse=True
        )
        neighbors[key] = ranked[:config.num_neighbors]

    with open(neighbors_path, "wb") as f:
        pickle.dump(neighbors, f)
    print(f"✅ Saved neighbors to {neighbors_path}")

if __name__ == "__main__":
    main()
