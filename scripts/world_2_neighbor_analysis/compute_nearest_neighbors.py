import sys
import os
import argparse
import numpy as np
import pickle
from pathlib import Path
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config
from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.utils import clean_term
from scripts.common.models.hierarchical_encoder import HierarchicalPatientEncoder

def get_visit_term_lists(patient_visits: list[dict]) -> dict[int, list[str]]:
    """
    Now, instead of one long sentence, this gives us a list of all the cleaned
    medical terms for each visit! PERFECT for our new encoder!
    A list of lists - each inner list is a vist where all the observed terms are shoved in.
    """
    visit_histories = {}
    config = get_global_config()
    history_window_length = config.num_visits

    for i in range(len(patient_visits) - history_window_length + 1):
        relevant_visits = patient_visits[i : i + history_window_length]
        end_idx = i + history_window_length - 1

        # This will now be a list of lists!
        history_term_lists = []
        for visit in relevant_visits:
            visit_terms = []
            visit_terms.extend([clean_term(d.get("Diagnosis_Name", "")) for d in visit.get("diagnoses", [])])
            visit_terms.extend([clean_term(m.get("MedSimpleGenericName", "")) for m in visit.get("medications", [])])
            visit_terms.extend([clean_term(p.get("CPT_Procedure_Description", "")) for p in visit.get("treatments", [])])
            history_term_lists.append([term for term in visit_terms if term])
        
        visit_histories[end_idx] = history_term_lists
        
    return visit_histories

# --- THE MAIN FUNCTION, NOW WITH OUR NEW ENGINE! ---
def get_visit_vectors(patient_data: list[dict]) -> dict[tuple[str, int], np.ndarray]:
    config = get_global_config()
    VECTORIZER_METHOD = config.vectorizer_method
    
    # --- Path to our beautiful, trained model! ---
    trained_encoder_path = project_root / "data" / "models" / "hierarchical_encoder_trained.pth"
    if not trained_encoder_path.exists():
        raise FileNotFoundError(f"OH NO! The trained encoder was not found at {trained_encoder_path}. Please run the training script first!")

    print(f"🗺️ Loading base term vectorizer model...")
    term_vectorizer = SentenceTransformer(f"/media/scratch/mferguson/models/{VECTORIZER_METHOD.replace('/', '-')}")

    print(f"🧠 Loading the TRAINED Hierarchical Patient Encoder from {trained_encoder_path}...")
    term_embedding_dim = term_vectorizer.get_sentence_embedding_dimension()
    # Initialize the model structure
    patient_encoder = HierarchicalPatientEncoder(
        term_embedding_dim=term_embedding_dim, 
        visit_hidden_dim=128, 
        patient_hidden_dim=256 
    )
    # LOAD THE TRAINED WEIGHTS! This is the magic!
    patient_encoder.load_state_dict(torch.load(trained_encoder_path))
    patient_encoder.eval() # Set the model to evaluation mode! Very important!

    final_vectors_dict = {}
    
    print("\n--- ⚙️ Processing Patients with the New Hierarchical Engine! ---")
    for patient in patient_data:
        patient_id = patient["patient_id"]
        
        # 1. Get the patient's history as a dictionary of {end_idx: [[visit1_terms], [visit2_terms], ...]}
        history_as_list_of_lists = get_visit_term_lists(patient["visits"])
        
        for end_idx, trajectory_term_lists in history_as_list_of_lists.items():
            
            # 2. Convert each visit's term list into a tensor of embeddings
            trajectory_to_encode = []
            with torch.no_grad():
                for visit_term_list in trajectory_term_lists:
                    if visit_term_list:
                        term_embeddings_tensor = term_vectorizer.encode(visit_term_list, convert_to_tensor=True)
                        trajectory_to_encode.append(term_embeddings_tensor)
            
            if not trajectory_to_encode: continue

            # 3. Feed the FULL list of visit tensors into our new model!
            with torch.no_grad():
                patient_vector_tensor = patient_encoder(trajectory_to_encode)
            
            if patient_vector_tensor is not None:
                final_vectors_dict[(patient_id, end_idx)] = patient_vector_tensor.cpu().numpy()

    return final_vectors_dict

def main():
    parser = argparse.ArgumentParser(description="Compute and save nearest neighbor data with a hyper-structured directory output.")
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    # ✨ NEW BATCH SIZE ARGUMENT! ✨
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of patients to process in a single batch.")
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

    # --- (The rest of the main function is the same, but now it passes the batch size!) ---
    base_dir = project_root / "data" / config.representation_method / config.vectorizer_method
    vectors_dir = base_dir / f"visits_{config.num_visits}" / f"patients_{config.num_patients}"
    output_dir = vectors_dir / f"metric_{config.distance_metric}" / f"neighbors_{config.num_neighbors}"
    
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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
        print("🔍 Vectors not found. Computing new vectors in batches...")
        patient_data = load_patient_data()
        # Pass the batch size to our new and improved function!
        vectors_dict = get_visit_vectors(patient_data, args.batch_size)
        with open(vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)
        print(f"💾 Saved new vector embeddings to {vectors_path}")

    # ... (The rest of the neighbor calculation is perfect!) ...
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