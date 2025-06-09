import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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

def get_visit_strings(patient_data: list[dict], window_len: int = 5) -> dict[str, dict[int, dict[tuple[str, int], str]]]:
    """
    For all patients, return a flat dictionary mapping:
        (patient_id, end_idx) → visit history string of the given window_len
    """
    result = {}
    for patient in patient_data:
        patient_id = patient["patient_id"]
        patient_visits = patient["visits"]
        visit_histories = get_visit_histories(patient_visits)

        if window_len not in visit_histories:
            continue  # Skip if this window length isn't valid for this patient

        for end_idx, visit_str in visit_histories[window_len].items():
            key = (patient_id, end_idx)
            result[key] = visit_str

    return result

######################################################################
# Use an sentence transformer to get vectors for each visit sequence
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import numpy as np

def get_visit_vectors(
    patient_data: list[dict],
    k: int,
    vectorizer: str = "sentence_transformer"
) -> dict[tuple[str, int], np.ndarray]:
    """
    Return TF-IDF vectors for each patient's visit sequences of length k.
    Keys are (patient_id, end_idx), values are np.ndarray vectors.
    """
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

    if vectorizer == "tfidf":
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
    elif vectorizer == "sentence_transformer":
        login(token="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")
        vectorizer_instance = SentenceTransformer("/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", local_files_only=True)
        # Encode the visit strings using the sentence transformer
        vectors = vectorizer_instance.encode(all_visit_strings, show_progress_bar=False, convert_to_numpy=True)
    else:
        assert False, f"Unknown vectorizer: {vectorizer}"

    return {key: vec for key, vec in zip(all_keys, vectors)}

def get_neighbors(patient_data, use_synthetic_data: bool=False, num_visits: int=5, distance_metric: str="cosine", vectorizer: str="sentence_transformer") -> dict[tuple[str, int], list[tuple[tuple[str, int], float]]]:
    """
    Compute and sort by closeness the neighbors for each visit sequence of length `num_visits`.

    Returns:
        dict mapping (patient_id, visit_idx) to list of (neighbor_id, similarity score)
    """
    try:
        with open(f"{'data' if use_synthetic_data else 'real_data'}/neighbors_{vectorizer}_{distance_metric}.pkl", "rb") as f:
            neighbors = pickle.load(f)
    except:
        vectors_dict = get_visit_vectors(patient_data, num_visits, vectorizer=vectorizer)
        with open(f"all_vectors_{vectorizer}_{num_visits}.pkl", "wb") as f:
            pickle.dump(vectors_dict, f)
        keys = list(vectors_dict.keys())
        matrix = np.vstack([vectors_dict[key] for key in keys])
        # Compute cosine similarity between every pair of visits, where sim_matrix[i][j] is the similarity between keys[i] and keys[j]
        if distance_metric == "euclidean":
            # Use negative distances to convert to similarity (higher is closer)
            sim_matrix = -euclidean_distances(matrix)
        elif distance_metric == "cosine":
            # Normalize the vectors to unit length before computing cosine similarity
            matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
            sim_matrix = cosine_similarity(matrix)
        else:
            assert False, f"Unknown distance metric: {distance_metric}"

        neighbors = {}

        for i, patient_by_visit in enumerate(keys):
            sims = sim_matrix[i]
            # Create a list of tuples - first element is tuple is patient/visit_idx pair, and second element is the cosine similarity between that patient/visit_idx pair and this one
            sim_pairs = [(keys[j], sims[j], vectors_dict[keys[j]]) for j in range(len(keys)) if j != i]
            # Sort by decreasing cosine similarity, because that's what's farther away
            sim_pairs.sort(key=lambda x: x[1], reverse=True)
            neighbors[patient_by_visit] = [pair for pair in sim_pairs]

        with open(f"{'synthetic_data' if use_synthetic_data else 'real_data'}/neighbors_{vectorizer}_{distance_metric}.pkl", "wb") as f:
            pickle.dump(neighbors, f)
        
    return neighbors