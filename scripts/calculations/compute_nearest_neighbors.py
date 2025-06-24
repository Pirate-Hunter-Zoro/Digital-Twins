import sys
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# --- Dynamic sys.path adjustment for module imports ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

from scripts.config import get_global_config
# --- FIX APPLIED: Import from our new central utility script! ---
from scripts.common.utils import turn_to_sentence

def get_visit_histories(patient_visits: list[dict]) -> dict[int, str]:
    """
    For ONE patient, returns a dictionary containing ONLY the MOST RECENT
    visit history window of length from the global config.
    """
    visit_histories = {}
    n = len(patient_visits)
    history_window_length = get_global_config().num_visits

    if n < history_window_length:
        return visit_histories

    start_idx = n - history_window_length
    end_idx = n - 1

    history = " | ".join(
        turn_to_sentence(patient_visits[i]) for i in range(start_idx, end_idx + 1)
    )
    
    visit_histories[end_idx] = history
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

    if get_global_config().vectorizer_method == "tfidf":
        vectorizer_instance = TfidfVectorizer()
        vectors = vectorizer_instance.fit_transform(all_visit_strings).toarray()
    elif get_global_config().vectorizer_method == "sentence_transformer":
        vectorizer_instance = SentenceTransformer("/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", local_files_only=True)
        
        batch_size = 32
        print(f"Encoding visit strings with batch size: {batch_size}")
        vectors = vectorizer_instance.encode(
            all_visit_strings,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    else:
        raise ValueError(f"Unknown vectorizer: {get_global_config().vectorizer_method}")

    return {key: vec for key, vec in zip(all_keys, vectors)}


def get_neighbors(patient_data) -> dict[tuple[str, int], list[tuple[tuple[str, int], float, np.ndarray]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence of length `num_visits`.
    """
    neighbors_file_path = f"real_data/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl"
    
    try:
        with open(neighbors_file_path, "rb") as f:
            print(f"Loading pre-computed neighbors from {neighbors_file_path}")
            neighbors = pickle.load(f)
    except FileNotFoundError:
        print("Pre-computed neighbors not found. Calculating from scratch...")
        vectors_dict = get_visit_vectors(patient_data)
        
        all_vectors_path = f"real_data/all_vectors_{get_global_config().vectorizer_method}_{get_global_config().num_visits}.pkl"
        with open(all_vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)
        print(f"Saved all visit vectors to {all_vectors_path}")

        keys = list(vectors_dict.keys())
        matrix = np.vstack(list(vectors_dict.values()))
        
        if get_global_config().distance_metric == "euclidean":
            sim_matrix = -euclidean_distances(matrix)
        elif get_global_config().distance_metric == "cosine":
            matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix_normalized = matrix / np.where(matrix_norm == 0, 1e-9, matrix_norm)
            sim_matrix = cosine_similarity(matrix_normalized)
        else:
            raise ValueError(f"Unknown distance metric: {get_global_config().distance_metric}")

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
