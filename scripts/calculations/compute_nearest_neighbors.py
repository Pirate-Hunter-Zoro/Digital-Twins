import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.config import get_global_config

#####################################################################
# Helper functions for turning visit data into strings

def turn_to_sentence(visit: dict) -> str:
    """
    Convert a visit dictionary to a human-readable sentence.
    """
    sentences = []
    if visit["diagnoses"]:
        sentences.append("Diagnoses: " + ", ".join(visit["diagnoses"]))
    if visit["medications"]:
        sentences.append("Medications: " + ", ".join(visit["medications"]))
    if visit["treatments"]:
        sentences.append("Treatments: " + ", ".join(visit["treatments"]))
    return "; ".join(sentences)

def get_visit_histories(patient_visits: list[dict]) -> dict[int, dict[int, str]]:
    """
    For ONE patient, returns a dictionary:
      Key = ending visit index (inclusive)
      Value = string representation of that window's visit history for the global config's window length.
    """
    visit_histories = {}

    n = len(patient_visits)
    if n < 1:
        return visit_histories  # No visits to process

    visit_histories = {}
    for start_idx in range(n - get_global_config().num_visits + 1):
        end_idx = start_idx + get_global_config().num_visits - 1
        history = " | ".join(
            turn_to_sentence(patient_visits[i]) for i in range(start_idx, end_idx + 1)
        )
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