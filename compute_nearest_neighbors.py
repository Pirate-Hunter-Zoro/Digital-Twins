import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

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
    For ONE patient, returns a nested dictionary:
      Outer key = window length (number of visits in the sequence)
      Inner key = ending visit index (inclusive)
      Value = string representation of that window's visit history

    Example:
      visit_histories[2][3] = summary of visits [2, 3]
    """
    visit_histories = {}

    n = len(patient_visits)
    if n < 1:
        return visit_histories  # No visits to process

    for window_length in range(1, n + 1):
        visit_histories[window_length] = {}
        for start_idx in range(n - window_length + 1):
            end_idx = start_idx + window_length - 1
            history = " | ".join(
                turn_to_sentence(patient_visits[i]) for i in range(start_idx, end_idx + 1)
            )
            visit_histories[window_length][end_idx] = history

    return visit_histories

def get_visit_strings(patient_data: list[dict]) -> dict[str, dict[int, dict[tuple[str, int], str]]]:
    """
    For all window sizes and all patients, return:
      {
        patient_id: {
          window_length: {
            (patient_id, end_idx): "visit history string",
            ...
          },
          ...
        },
        ...
      }
    """
    result = {}
    for patient in patient_data:
        patient_id = patient["patient_id"]
        patient_visits = patient["visits"]
        visit_histories = get_visit_histories(patient_visits)
        
        # Convert: {window_len: {end_idx: string}} → {window_len: {(pid, end_idx): string}}
        result[patient_id] = {
            window_len: {
                (patient_id, end_idx): visit_str
                for end_idx, visit_str in windows.items()
            }
            for window_len, windows in visit_histories.items()
        }

    return result

######################################################################
# Initialize the TF-IDF vectorizer to turn visit strings into vectors

vectorizer = TfidfVectorizer()

def get_visit_vectors(
    patient_data: list[dict],
    k: int,
    vectorizer=None
) -> dict[tuple[str, int], np.ndarray]:
    """
    Return TF-IDF vectors for each patient's visit sequences of length k.
    Keys are (patient_id, end_idx), values are np.ndarray vectors.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    all_visit_strings = []
    all_keys = []

    for patient in patient_data:
        patient_id = patient["patient_id"]
        patient_visits = patient["visits"]
        visit_histories = get_visit_histories(patient_visits)

        if k not in visit_histories:
            continue

        for end_idx, visit_string in visit_histories[k].items():
            all_keys.append((patient_id, end_idx))
            all_visit_strings.append(visit_string)

    vectors = vectorizer.fit_transform(all_visit_strings)
    dense_vectors = vectors.toarray()

    return {key: vec for key, vec in zip(all_keys, dense_vectors)}

def get_neighbors(patient_data, num_visits: int=5) -> dict[tuple[str, int], list[tuple[tuple[str, int], float]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence of length `num_visits`.

    Returns:
        dict mapping (patient_id, visit_idx) to list of (neighbor_id, similarity score)
    """
    try:
        with open("neighbors.pkl", "rb") as f:
            neighbors = pickle.load(f)
    except:
        vectors_dict = get_visit_vectors(patient_data, num_visits)
        keys = list(vectors_dict.keys())
        matrix = np.vstack([vectors_dict[key] for key in keys])
        # Compute cosine similarity between every pair of visits, where sim_matrix[i][j] is the similarity between keys[i] and keys[j]
        sim_matrix = cosine_similarity(matrix)

        neighbors = {}

        for i, patient_by_visit in enumerate(keys):
            sims = sim_matrix[i]
            # Create a list of tuples - first element is tuple is patient/visit_idx pair, and second element is the cosine similarity between that patient/visit_idx pair and this one
            sim_pairs = [(keys[j], sims[j]) for j in range(len(keys)) if j != i]
            # Sort by decreasing cosine similarity, because that's what's farther away
            sim_pairs.sort(key=lambda x: x[1], reverse=True)
            neighbors[patient_by_visit] = [pair for pair in sim_pairs]

        with open("neighbors.pkl", "wb") as f:
            pickle.dump(neighbors, f)
        
    return neighbors