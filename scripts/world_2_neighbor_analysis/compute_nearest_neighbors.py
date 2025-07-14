import sys
import os
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.utils import turn_to_sentence
from scripts.common.data.load_patient_data import load_patient_data     

def get_visit_histories(patient_visits: list[dict]) -> dict[int, str]:
    visit_histories = {}
    config = get_global_config()
    history_window_length = config.num_visits

    for i in range(len(patient_visits) - history_window_length + 1):
        relevant_visits = patient_visits[i : i + history_window_length]
        end_idx = i + history_window_length - 1

        if config.representation_method == "visit_sentence":
            history_string = " | ".join(turn_to_sentence(visit) for visit in relevant_visits)
        elif config.representation_method == "bag_of_codes":
            all_codes = []
            for visit in relevant_visits:
                all_codes.extend([d.get("Diagnosis_Name") for d in visit.get("diagnoses", []) if d.get("Diagnosis_Name")])
                all_codes.extend([m.get("MedSimpleGenericName") for m in visit.get("medications", []) if m.get("MedSimpleGenericName")])
                all_codes.extend([p.get("CPT_Procedure_Description") for p in visit.get("treatments", []) if p.get("CPT_Procedure_Description")])
            history_string = " ".join(filter(None, all_codes))
        else:
            raise ValueError(f"Unknown representation method: {config.representation_method}")

        visit_histories[end_idx] = history_string
        
    return visit_histories


def get_visit_vectors(patient_data: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    all_visit_strings, all_keys = [], []
    for patient in patient_data:
        visit_histories = get_visit_histories(patient["visits"])
        for end_idx, visit_string in visit_histories.items():
            all_keys.append((patient["patient_id"], end_idx))
            all_visit_strings.append(visit_string)

    vectorizer_path = project_root / "models" / get_global_config().vectorizer_method
    vectorizer_instance = SentenceTransformer(str(vectorizer_path), local_files_only=True)

    print(f"Encoding {len(all_visit_strings)} visit strings...")
    vectors = vectorizer_instance.encode(all_visit_strings, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return {key: vec for key, vec in zip(all_keys, vectors)}

def main():
    parser = argparse.ArgumentParser(description="Compute and save nearest neighbor data with a hyper-structured directory output.")
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    config = get_global_config()

    # --- ✨ NEW: Build the hyper-structured directories ---
    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    output_dir = vectors_dir / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Vector directory: {vectors_dir}")
    print(f"📂 Output directory: {output_dir}")

    neighbors_path = output_dir / "neighbors.pkl"
    vectors_path = vectors_dir / "all_vectors.pkl"

    if os.path.exists(neighbors_path):
        print(f"✅ Neighbors already computed at {neighbors_path}. Skipping.")
        return

    if os.path.exists(vectors_path):
        print(f"✅ Loading existing vectors from {vectors_path}")
        with open(vectors_path, "rb") as f:
            vectors_dict = pickle.load(f)
    else:
        print("🔍 Vectors not found. Computing new vectors...")
        patient_data = load_patient_data()
        vectors_dict = get_visit_vectors(patient_data)
        with open(vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)
        print(f"💾 Saved new vector embeddings to {vectors_path}")

    print("🔍 Computing nearest neighbors with ✨Neighbor Diversity Filter✨...")
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
        patient_id_of_interest, _ = key
        
        sims = sim_matrix[i]
        ranked = sorted(
            [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if keys[j][0] != patient_id_of_interest],
            key=lambda x: x[1],
            reverse=True
        )

        filtered_neighbors = []
        seen_neighbor_patients = set()
        for neighbor_tuple in ranked:
            neighbor_key, _, _ = neighbor_tuple
            neighbor_patient_id, _ = neighbor_key
            if neighbor_patient_id not in seen_neighbor_patients:
                filtered_neighbors.append(neighbor_tuple)
                seen_neighbor_patients.add(neighbor_patient_id)
            if len(filtered_neighbors) >= config.num_neighbors:
                break
        
        neighbors[key] = filtered_neighbors

    with open(neighbors_path, "wb") as f:
        pickle.dump(neighbors, f)
    print(f"✅ Saved diverse neighbors to {neighbors_path}")

if __name__ == "__main__":
    main()