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
from scripts.common.models.hierarchical_encoder import HierarchicalPatientEncoder
# ✨ We now import our helper from the common workshop! ✨
from scripts.common.utils import get_visit_term_lists 

def get_visit_vectors(patient_data: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    config = get_global_config()
    VECTORIZER_METHOD = config.vectorizer_method
    
    trained_encoder_path = project_root / "data" / "models" / "hierarchical_encoder_trained.pth"
    if not trained_encoder_path.exists():
        raise FileNotFoundError(f"OH NO! The trained encoder was not found at {trained_encoder_path}. Please run the training script first!")

    print(f"🗺️ Loading base term vectorizer model...")
    term_vectorizer = SentenceTransformer(f"/media/scratch/mferguson/models/{VECTORIZER_METHOD.replace('/', '-')}")

    print(f"🧠 Loading the TRAINED Hierarchical Patient Encoder from {trained_encoder_path}...")
    term_embedding_dim = term_vectorizer.get_sentence_embedding_dimension()
    patient_encoder = HierarchicalPatientEncoder(
        term_embedding_dim=term_embedding_dim, visit_hidden_dim=128, patient_hidden_dim=256 
    )
    patient_encoder.load_state_dict(torch.load(trained_encoder_path))
    patient_encoder.eval()

    final_vectors_dict = {}
    
    print("\n--- ⚙️ Processing Patients with the New Hierarchical Engine! ---")
    for patient in patient_data:
        patient_id = patient["patient_id"]
        
        # ✨ CORRECTED! Calling the right function from our utils! ✨
        history_as_list_of_lists = get_visit_term_lists(patient["visits"], config.num_visits)
        
        for end_idx, trajectory_term_lists in history_as_list_of_lists.items():
            trajectory_to_encode = []
            with torch.no_grad():
                for visit_term_list in trajectory_term_lists:
                    if visit_term_list:
                        term_embeddings_tensor = term_vectorizer.encode(visit_term_list, convert_to_tensor=True)
                        trajectory_to_encode.append(term_embeddings_tensor)
            
            if not trajectory_to_encode: continue

            with torch.no_grad():
                patient_vector_tensor = patient_encoder(trajectory_to_encode)
            
            if patient_vector_tensor is not None:
                final_vectors_dict[(patient_id, end_idx)] = patient_vector_tensor.cpu().numpy()

    return final_vectors_dict

def main():
    # We're adding all the expected arguments right here!
    parser = argparse.ArgumentParser(description="Compute and save ALL ranked neighbors using the hierarchical encoder.")
    parser.add_argument("--representation_method", default="visit_sentence")
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="cosine")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    args = parser.parse_args()

    # Now we use the arguments to set up our config!
    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors
    )
    config = get_global_config()

    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method.replace('/', '-')
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    output_dir = vectors_dir / f"metric_{config.distance_metric}"
    
    os.makedirs(output_dir, exist_ok=True)

    neighbors_path = output_dir / "all_ranked_neighbors.pkl"
    vectors_path = vectors_dir / "all_vectors.pkl"

    if os.path.exists(neighbors_path):
        print(f"✅ All ranked neighbors already computed at {neighbors_path}. Skipping.")
        return

    if os.path.exists(vectors_path):
        with open(vectors_path, "rb") as f: vectors_dict = pickle.load(f)
    else:
        patient_data = load_patient_data()
        # ✨ CORRECTED! The function now only needs one argument! ✨
        vectors_dict = get_visit_vectors(patient_data)
        with open(vectors_path, "wb") as f: pickle.dump(vectors_dict, f)

    print("🔍 Computing and ranking ALL neighbors for every patient...")
    keys = list(vectors_dict.keys())
    matrix = np.vstack([vectors_dict[k] for k in keys])
    sim_matrix = cosine_similarity(matrix)

    all_ranked_neighbors = {}
    for i, key in enumerate(keys):
        sims = sim_matrix[i]
        potential_neighbors = [(keys[j], sims[j]) for j in range(len(keys)) if i != j]
        ranked_list = sorted(potential_neighbors, key=lambda x: x[1], reverse=True)
        all_ranked_neighbors[key] = ranked_list

    with open(neighbors_path, "wb") as f:
        pickle.dump(all_ranked_neighbors, f)
    print(f"✅ Saved the Flexible Behemoth (all ranked neighbors) to {neighbors_path}")

if __name__ == "__main__":
    main()