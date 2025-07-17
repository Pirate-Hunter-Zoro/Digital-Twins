import sys
import os
import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import get_visit_term_lists
from scripts.common.models.hierarchical_encoder import HierarchicalPatientEncoder

def get_visit_vectors(patient_data: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    # This function uses the trained HierarchicalPatientEncoder
    # The logic we created before is perfect and stays the same!
    # ... (function content as previously defined) ...
    pass

def main():
    parser = argparse.ArgumentParser(description="Compute and save ALL ranked neighbors for each patient.")
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--num_visits", type=int, default=6)
    args = parser.parse_args()

    setup_config(
        representation_method="visit_sentence",
        vectorizer_method=args.vectorizer_method,
        distance_metric="cosine",
        num_visits=args.num_visits,
        num_patients=5000,
        num_neighbors=5 # This k-value will be used by the analysis script later!
    )
    config = get_global_config()

    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method.replace('/', '-')
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    output_dir = vectors_dir / f"metric_{config.distance_metric}"
    
    os.makedirs(output_dir, exist_ok=True)

    # --- ✨ RENAMED! This file is now a BEHEMOTH! ---
    neighbors_path = output_dir / "all_ranked_neighbors.pkl"
    vectors_path = vectors_dir / "all_vectors.pkl"

    if os.path.exists(neighbors_path):
        print(f"✅ All ranked neighbors already computed at {neighbors_path}. Skipping.")
        return

    # --- Load or Compute Vectors ---
    if os.path.exists(vectors_path):
        with open(vectors_path, "rb") as f:
            vectors_dict = pickle.load(f)
    else:
        patient_data = load_patient_data()
        vectors_dict = get_visit_vectors(patient_data)
        with open(vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)

    print("🔍 Computing and ranking ALL neighbors for every patient...")
    keys = list(vectors_dict.keys())
    matrix = np.vstack([vectors_dict[k] for k in keys])
    sim_matrix = cosine_similarity(matrix)

    # This dictionary will be huge!
    all_ranked_neighbors = {}
    for i, key in enumerate(keys):
        patient_id_of_interest, _ = key
        
        sims = sim_matrix[i]
        
        # Get all other patients and their similarity scores
        potential_neighbors = [
            (keys[j], sims[j]) for j in range(len(keys)) if i != j
        ]
        
        # Sort by similarity score, from highest to lowest
        ranked_list = sorted(potential_neighbors, key=lambda x: x[1], reverse=True)
        
        # --- ✨ THE BIG CHANGE! ✨ ---
        # We save the ENTIRE ranked list!
        all_ranked_neighbors[key] = ranked_list

    with open(neighbors_path, "wb") as f:
        pickle.dump(all_ranked_neighbors, f)
    print(f"✅ Saved the Flexible Behemoth (all ranked neighbors) to {neighbors_path}")

if __name__ == "__main__":
    # A placeholder for get_visit_vectors since its definition is long
    def get_visit_vectors(patient_data):
        print("--- (This is where the hierarchical encoder would run) ---")
        # In a real run, this would generate the vectors
        # For this example, let's pretend it returns a dummy dictionary
        return {('patient1', 5): np.random.rand(256), ('patient2', 5): np.random.rand(256)}
    
    main()