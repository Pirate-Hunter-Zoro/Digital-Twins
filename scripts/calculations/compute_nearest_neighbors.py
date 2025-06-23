import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.config import get_global_config # Ensure this is correct

#####################################################################
# Helper functions for turning visit data into strings

# This function is in scripts/calculations/compute_nearest_neighbors.py
def turn_to_sentence(encounter_obj: dict) -> str:
    """
    Convert an encounter dictionary (from real EHR data) into a human-readable sentence,
    extracting specific string fields from nested dictionaries for diagnoses, medications, and treatments (procedures).
    """
    sentences = []
    
    # Diagnoses: Extract Diagnosis_1_Code from each diagnosis dictionary
    # encounter_obj["diagnoses"] is a list of dicts. We need the 'Diagnosis_1_Code' from each dict.
    if encounter_obj.get("diagnoses"):
        diagnoses_codes = [
            diag.get("Diagnosis_1_Code") # Get the code from the dictionary
            for diag in encounter_obj["diagnoses"] 
            if diag.get("Diagnosis_1_Code") is not None # Ensure the code exists
        ]
        if diagnoses_codes: # Only append if there are actual codes to join
            sentences.append("Diagnoses: " + ", ".join(diagnoses_codes))

    # Medications: Extract MedName from each medication dictionary
    # encounter_obj["medications"] is a list of dicts. We need the 'MedName' from each dict.
    if encounter_obj.get("medications"):
        medication_names = [
            med.get("MedName") 
            for med in encounter_obj["medications"] 
            if med.get("MedName") is not None
        ]
        if medication_names:
            sentences.append("Medications: " + ", ".join(medication_names))
    
    # Treatments: Extract Procedure_Description from each procedure dictionary
    # encounter_obj["treatments"] (which was mapped from 'procedures' in load_patient_data.py) is a list of dicts.
    # We need the 'Procedure_Description' from each dict.
    if encounter_obj.get("treatments"): 
        treatment_descriptions = [
            proc.get("Procedure_Description") 
            for proc in encounter_obj["treatments"] 
            if proc.get("Procedure_Description") is not None # Ensure description exists
        ]
        if treatment_descriptions:
            sentences.append("Treatments: " + ", ".join(treatment_descriptions))

    return "; ".join(sentences)

######################################################################
# Use an sentence transformer to get vectors for each visit sequence

def get_visit_histories(patient_visits: list[dict]) -> dict[int, str]:
    """
    For ONE patient, returns a dictionary containing ONLY the MOST RECENT
    visit history window of length from the global config.
    """
    visit_histories = {}
    n = len(patient_visits)
    history_window_length = get_global_config().num_visits

    # Patient needs at least 'history_window_length' visits to form one history.
    if n < history_window_length:
        return visit_histories # Not enough visits, return empty.
    
    # The single most recent window starts at the end.
    start_idx = n - history_window_length
    end_idx = n - 1 # The last visit in the list

    # Generate the single history string for this one window.
    history = " | ".join(
        turn_to_sentence(patient_visits[i]) for i in range(start_idx, end_idx + 1)
    )
    
    # The key is the index of the last visit in this history window.
    visit_histories[end_idx] = history

    return visit_histories

def get_visit_strings(patient_data: list[dict]) -> dict[str, dict[int, dict[tuple[str, int], str]]]:
    """
    For all patients, return a flat dictionary mapping:
        (patient_id, end_idx) → visit history string of the given window_len
    """
    result = {}
    for patient in patient_data:
        patient_id = patient["patient_id"]
        patient_visits = patient["visits"]
        visit_histories = get_visit_histories(patient_visits)

        for end_idx, visit_str in visit_histories.items():
            key = (patient_id, end_idx)
            result[key] = visit_str

    return result

######################################################################
# Use an sentence transformer to get vectors for each visit sequence
from sentence_transformers import SentenceTransformer
import numpy as np

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
        vectorizer_instance = TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),  # Unigrams and bigrams
                                max_features=10000,  # Limit to 10,000 features
                                stop_words="english",  # Remove common English stop words
                                lowercase=True,  # Convert to lowercase
                                norm="l2",  # Normalize the vectors
                                use_idf=True,  # Use inverse document frequency
                                smooth_idf=True,  # Smooth IDF to avoid division by zero
                                sublinear_tf=True,  # Use sublinear term frequency scaling
                            ),
        vectors = vectorizer_instance.fit_transform(all_visit_strings).toarray()
    elif get_global_config().vectorizer_method == "sentence_transformer":
        vectorizer_instance = SentenceTransformer("/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", local_files_only=True)
        # Encode the visit strings using the sentence transformer
        vectors = vectorizer_instance.encode(all_visit_strings, show_progress_bar=False, convert_to_numpy=True)
    else:
        assert False, f"Unknown vectorizer: {get_global_config().vectorizer_method}"

    return {key: vec for key, vec in zip(all_keys, vectors)}

def get_neighbors(patient_data) -> dict[tuple[str, int], list[tuple[tuple[str, int], float]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence of length `num_visits`.

    Returns:
        dict mapping (patient_id, visit_idx) to list of (neighbor_id, similarity score)
    """
    try:
        with open(f"real_data/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl", "rb") as f:
            neighbors = pickle.load(f)
    except:
        vectors_dict = get_visit_vectors(patient_data)
        with open(f"real_data/all_vectors_{get_global_config().vectorizer_method}_{get_global_config().num_visits}.pkl", "wb") as f:
            pickle.dump(vectors_dict, f)
        keys = list(vectors_dict.keys())
        matrix = np.vstack([vectors_dict[key] for key in keys])
        # Compute cosine similarity between every pair of visits, where sim_matrix[i][j] is the similarity between keys[i] and keys[j]
        if get_global_config().distance_metric == "euclidean":
            # Use negative distances to convert to similarity (higher is closer)
            sim_matrix = -euclidean_distances(matrix)
        elif get_global_config().distance_metric == "cosine":
            # Normalize the vectors to unit length before computing cosine similarity
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
            sim_matrix = cosine_similarity(matrix)
        else:
            assert False, f"Unknown distance metric: {get_global_config().distance_metric}"

        neighbors = {}

        for i, patient_by_visit in enumerate(keys):
            sims = sim_matrix[i]
            # Create a list of tuples - first element is tuple is patient/visit_idx pair, and second element is the cosine similarity between that patient/visit_idx pair and this one
            sim_pairs = [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if j != i]
            # Sort by decreasing cosine similarity, because that's what's farther away
            sim_pairs.sort(key=lambda x: x[1], reverse=True)
            neighbors[patient_by_visit] = [pair for pair in sim_pairs]

        with open(f"real_data/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl", "wb") as f:
            pickle.dump(neighbors, f)
        
    return neighbors