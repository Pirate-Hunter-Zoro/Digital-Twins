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

# This function is in scripts/calculations/compute_nearest_neighbors.py
def turn_to_sentence(encounter_obj: dict) -> str:
    """
    Convert an encounter dictionary (from real EHR data) into a human-readable sentence,
    extracting specific string fields from nested dictionaries for diagnoses, medications, and treatments (procedures).
    """
    sentences = []
    
    if encounter_obj.get("diagnoses"):
        diagnoses_codes = [
            diag.get("Diagnosis_Name")
            for diag in encounter_obj["diagnoses"] 
            if diag.get("Diagnosis_Name") is not None
        ]
        if diagnoses_codes:
            sentences.append("Diagnoses: " + ", ".join(diagnoses_codes))

    if encounter_obj.get("medications"):
        medication_names = [
            med.get("MedSimpleGenericName")
            for med in encounter_obj["medications"] 
            if med.get("MedSimpleGenericName") is not None
        ]
        if medication_names:
            sentences.append("Medications: " + ", ".join(medication_names))
    
    if encounter_obj.get("treatments"): 
        treatment_descriptions = [
            proc.get("CPT_Procedure_Description")
            for proc in encounter_obj["treatments"] 
            if proc.get("CPT_Procedure_Description") is not None
        ]
        if treatment_descriptions:
            sentences.append("Treatments: " + ", ".join(treatment_descriptions))

    return "; ".join(sentences)


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
    --- MODIFIED TO PREVENT GPU POWER OVERLOAD ---
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
        vectorizer_instance = TfidfVectorizer() # Simplified for example
        vectors = vectorizer_instance.fit_transform(all_visit_strings).toarray()
    elif get_global_config().vectorizer_method == "sentence_transformer":
        vectorizer_instance = SentenceTransformer("/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", local_files_only=True)
        
        # --- FIX APPLIED: Added batch_size to regulate GPU memory usage! ---
        # This tells the encoder to process the data in small, manageable chunks
        # instead of all at once. No more power surges!
        batch_size = 32
        print(f"Encoding visit strings with batch size: {batch_size}")
        vectors = vectorizer_instance.encode(
            all_visit_strings,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # --- End of Fix ---
    else:
        assert False, f"Unknown vectorizer: {get_global_config().vectorizer_method}"

    return {key: vec for key, vec in zip(all_keys, vectors)}


def get_neighbors(patient_data) -> dict[tuple[str, int], list[tuple[tuple[str, int], float, np.ndarray]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence of length `num_visits`.
    Returns a dictionary mapping a visit to its neighbors, including similarity scores and vectors.
    """
    neighbors_file_path = f"real_data/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl"
    
    try:
        with open(neighbors_file_path, "rb") as f:
            print(f"Loading pre-computed neighbors from {neighbors_file_path}")
            neighbors = pickle.load(f)
    except FileNotFoundError:
        print("Pre-computed neighbors not found. Calculating from scratch...")
        vectors_dict = get_visit_vectors(patient_data)
        
        # Save the vectors for potential reuse
        all_vectors_path = f"real_data/all_vectors_{get_global_config().vectorizer_method}_{get_global_config().num_visits}.pkl"
        with open(all_vectors_path, "wb") as f:
            pickle.dump(vectors_dict, f)
        print(f"Saved all visit vectors to {all_vectors_path}")

        keys = list(vectors_dict.keys())
        matrix = np.vstack(list(vectors_dict.values()))
        
        if get_global_config().distance_metric == "euclidean":
            # Use negative distances for similarity - since we'll be sorting in reverse order shortly
            sim_matrix = -euclidean_distances(matrix)
        elif get_global_config().distance_metric == "cosine":
            matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix_normalized = matrix / np.where(matrix_norm == 0, 1e-9, matrix_norm) # Avoid division by zero
            sim_matrix = cosine_similarity(matrix_normalized)
        else:
            raise ValueError(f"Unknown distance metric: {get_global_config().distance_metric}")

        neighbors = {}
        for i, patient_by_visit in enumerate(keys):
            sims = sim_matrix[i]
            sim_pairs = [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if j != i]
            sim_pairs.sort(key=lambda x: x[1], reverse=True) # Higher similarity is better
            neighbors[patient_by_visit] = sim_pairs

        with open(neighbors_file_path, "wb") as f:
            pickle.dump(neighbors, f)
        print(f"Saved newly computed neighbors to {neighbors_file_path}")
        
    return neighbors