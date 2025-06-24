# --- compute_nearest_neighbors.py ---
import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from sentence_transformers import SentenceTransformer

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.config import get_global_config
from scripts.common.utils import turn_to_sentence

def get_visit_histories(patient_visits: list[dict]) -> dict[int, str]:
    """
    For ONE patient, returns a dictionary containing ONLY the MOST RECENT
    visit history window, formatted according to the configured representation method.
    --- THIS IS OUR NEW UPGRADE SLOT! ---
    """
    visit_histories = {}
    config = get_global_config()
    history_window_length = config.num_visits
    representation_method = config.representation_method

    if len(patient_visits) < history_window_length:
        return visit_histories

    # We only care about the latest sequence of visits for prediction
    start_idx = len(patient_visits) - history_window_length
    relevant_visits = patient_visits[start_idx:]
    end_idx = len(patient_visits) - 1

    history_string = ""
    if representation_method == "visit_sentence":
        # R3: Temporal "Visit-Sentence" method (our original method)
        history_string = " | ".join(turn_to_sentence(visit) for visit in relevant_visits)
    
    elif representation_method == "bag_of_codes":
        # R2: Unordered Bag-of-codes method
        all_codes = []
        for visit in relevant_visits:
            all_codes.extend([d.get("Diagnosis_Name") for d in visit.get("diagnoses", []) if d.get("Diagnosis_Name")])
            all_codes.extend([m.get("MedSimpleGenericName") for m in visit.get("medications", []) if m.get("MedSimpleGenericName")])
            all_codes.extend([p.get("CPT_Procedure_Description") for p in visit.get("treatments", []) if p.get("CPT_Procedure_Description")])
        history_string = " ".join(filter(None, all_codes))

    else:
        raise ValueError(f"Unknown representation method configured: {representation_method}")

    visit_histories[end_idx] = history_string
    return visit_histories


def get_visit_vectors(
    patient_data: list[dict],
) -> dict[tuple[str, int], np.ndarray]:
    """
    Keys are (patient_id, end_idx), values are np.ndarray vectors.
    """
    all_visit_strings = []
    all_keys = []

    for patient in patient_data:
        patient_id = patient["patient_id"]
        patient_visits = patient["visits"]
        visit_histories = get_visit_histories(patient_visits)

        for end_idx, visit_string in visit_histories.items():
            all_keys.append((patient_id, end_idx))
            all_visit_strings.append(visit_string)

    if get_global_config().vectorizer_method != "sentence_transformer":
         raise ValueError("Experiment A requires 'sentence_transformer' as the vectorizer method.")

    vectorizer_instance = SentenceTransformer("/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", local_files_only=True)
    
    batch_size = 32
    print(f"Encoding visit strings with batch size: {batch_size}")
    vectors = vectorizer_instance.encode(
        all_visit_strings,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return {key: vec for key, vec in zip(all_keys, vectors)}


def get_neighbors(patient_data) -> dict[tuple[str, int], list[tuple[tuple[str, int], float, np.ndarray]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence.
    Now includes representation_method in the cache file path to prevent data contamination between experiments!
    """
    config = get_global_config()
    # Updated to include representation method in filename!
    neighbors_file_path = f"real_data/neighbors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}.pkl"
    
    try:
        with open(neighbors_file_path, "rb") as f:
            print(f"Loading pre-computed neighbors from {neighbors_file_path}")
            neighbors = pickle.load(f)
    except FileNotFoundError:
        print("Pre-computed neighbors not found. Calculating from scratch...")
        vectors_dict = get_visit_vectors(patient_data)
        
        all_vectors_path = f"real_data/all_vectors_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}.pkl"
        with open(all_vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)
        print(f"Saved all visit vectors to {all_vectors_path}")

        keys = list(vectors_dict.keys())
        matrix = np.vstack(list(vectors_dict.values()))
        
        if config.distance_metric == "euclidean":
            sim_matrix = -euclidean_distances(matrix) 
        elif config.distance_metric == "cosine":
            sim_matrix = cosine_similarity(matrix)
        else:
            raise ValueError(f"Unknown distance metric: {config.distance_metric}")

        neighbors = {}
        for i, patient_by_visit in enumerate(keys):
            sims = sim_matrix[i]
            sim_pairs = [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if j != i]
            sim_pairs.sort(key=lambda x: x[1], reverse=True)
            neighbors[patient_by_visit] = sim_pairs

        with open(neighbors_file_path, "wb") as f:
            pickle.dump(neighbors, f)
        print(f"Saved newly computed neighbors to {neighbors_file_path}")
        
    return neighbors