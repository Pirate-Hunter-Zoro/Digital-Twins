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

def get_visit_histories(patient_visits: list[dict]) -> dict[int, dict[int, str]]:
    """
    For ONE patient, returns a dictionary:
      Key = ending visit index (inclusive)
      Value = string representation of that window's visit history for the global config's window length.
    """
    visit_histories = {}

    n = len(patient_visits)
    # The minimum number of visits for history is num_visits.
    # So if patient has fewer visits than num_visits, we cannot form a history window of that length.
    if n < get_global_config().num_visits: # Adjust this check for num_visits from config
        return visit_histories  # Not enough visits to form a history window

    # The loop should go up to n, so that the last window starts at n - num_visits.
    # If num_visits is 5, and patient has 5 visits (0-4), start_idx = 0.
    # end_idx = 0 + 5 - 1 = 4. range(0, 5-5+1 = 1) -> 0. so range(0,1) gives start_idx 0.
    # This means end_idx is based on the LAST visit in the window,
    # and the window itself is length 'get_global_config().num_visits'.
    # This logic assumes num_visits is the full window length, not history length + 1.
    # Let's adjust based on typical interpretation: num_visits is the history length.
    # If num_visits is 5, it means last 5 visits form the history.
    # For predicting visit N, we use visits up to N-1.
    # The config has num_visits for history.
    
    # Let's re-confirm config's num_visits purpose.
    # In generate_prompt, it uses `patient["visits"][:get_global_config().num_visits-1]`
    # So num_visits IS the actual length of the history window.
    # And get_global_config().num_visits-1 is the ENDING index of the history.
    # This means `num_visits` is the count of visits in history.

    # If get_global_config().num_visits = 5, we want to predict visit 5, using history visits 0-4.
    # The history should be patient["visits"][:get_global_config().num_visits]
    # The loop should identify all possible history windows of length num_visits.

    # Re-evaluate get_visit_histories logic with num_visits = history length
    # A patient needs AT LEAST get_global_config().num_visits + 1 visits to predict one next visit.
    # If num_visits=5, they need 6 visits to predict the 6th.
    
    # Let's assume num_visits is the length of the history window.
    # The loop should iterate over possible *prediction points*.
    # If num_visits=5, and patient has 10 visits:
    # Predict V5: history V0-V4.
    # Predict V6: history V1-V5.
    # ...
    # Predict V9: history V4-V8.

    # This function is used to create *candidate pool* visit strings.
    # Keys are (patient_id, end_idx) - where end_idx is the *last visit in the history window*.

    # For each visit that could be a 'next visit' to predict (from index `num_visits` onwards)
    # The history window ending at `end_idx` has length `get_global_config().num_visits`.
    # So `start_idx` should be `end_idx - get_global_config().num_visits + 1`.

    # Example: num_visits = 5
    # If patient_visits is [V0, V1, V2, V3, V4, V5, V6, V7] (length 8)
    # First history window: V0, V1, V2, V3, V4 (length 5). end_idx = 4. start_idx = 0.
    # Second history window: V1, V2, V3, V4, V5 (length 5). end_idx = 5. start_idx = 1.
    # Last history window: V3, V4, V5, V6, V7 (length 5). end_idx = 7. start_idx = 3.

    # The loop for `start_idx` needs to range from 0 up to `n - get_global_config().num_visits`.
    # `end_idx` will be `start_idx + get_global_config().num_visits - 1`.
    
    # `n - get_global_config().num_visits + 1` should be the number of possible history windows.
    # If n=5, num_visits=5, then 5-5+1 = 1 window (0-4). So start_idx = 0. This is fine.

    # The `turn_to_sentence` call will now work with the correct `encounter_obj` format.
    # The structure of this function seems okay for creating history strings based on `num_visits` window.
    # It seems to be correct.

    visit_histories = {}
    n = len(patient_visits)
    history_window_length = get_global_config().num_visits

    # Patient needs at least 'history_window_length' visits to form a history.
    if n < history_window_length:
        return visit_histories

    # Iterate over possible starting points of the history window
    # The loop for start_idx needs to go up to (n - history_window_length)
    # Example: n=5, history_window_length=5. range(0, 5-5+1 = 1). start_idx = 0.
    # Example: n=8, history_window_length=5. range(0, 8-5+1 = 4). start_idx = 0,1,2,3.
    # start_idx 0: history V0-V4, end_idx=4
    # start_idx 3: history V3-V7, end_idx=7
    for start_idx in range(n - history_window_length + 1):
        end_idx = start_idx + history_window_length - 1
        history = " | ".join(
            turn_to_sentence(patient_visits[i]) for i in range(start_idx, end_idx + 1)
        )
        visit_histories[end_idx] = history # Key is the index of the last visit in this history window

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