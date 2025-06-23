import sys
import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

# A configurable threshold for how similar two terms must be to be considered a "match"
SIMILARITY_THRESHOLD = 0.7 

def calculate_weighted_score(predicted_terms: list, actual_terms: list, idf_registry: dict, embedding_library: dict) -> float:
    """
    The core combat technique. Calculates a normalized, weighted score for a single category.
    """
    if not predicted_terms or not actual_terms:
        return 0.0

    # Get the Innate Techniques (embeddings) for all involved terms
    pred_vectors = np.array([embedding_library.get(term, np.zeros(768)) for term in predicted_terms])
    actual_vectors = np.array([embedding_library.get(term, np.zeros(768)) for term in actual_terms])

    # --- Greedy Best-First Matching ---
    # Calculate similarities between every predicted term and every actual term
    sim_matrix = cosine_similarity(pred_vectors, actual_vectors)
    
    successful_matches = []
    # Keep track of which actual terms have already been claimed by a prediction
    claimed_actual_indices = set()

    for i, pred_term in enumerate(predicted_terms):
        best_match_idx = -1
        highest_sim = -1.0

        # Find the best available partner for the current predicted term
        for j, actual_term in enumerate(actual_terms):
            if j not in claimed_actual_indices and sim_matrix[i, j] > highest_sim:
                highest_sim = sim_matrix[i, j]
                best_match_idx = j
        
        # If the best match is good enough, claim it!
        if best_match_idx != -1 and highest_sim >= SIMILARITY_THRESHOLD:
            matched_actual_term = actual_terms[best_match_idx]
            successful_matches.append((pred_term, matched_actual_term, highest_sim))
            claimed_actual_indices.add(best_match_idx)

    # --- Calculate the final score based on the successful matches ---
    raw_score = 0.0
    for pred_term, actual_term, similarity in successful_matches:
        idf_pred = idf_registry.get(pred_term, 0.0)
        idf_actual = idf_registry.get(actual_term, 0.0)
        # The geometric mean of the Cursed Energy levels
        rarity_score = np.sqrt(idf_pred * idf_actual) if idf_pred > 0 and idf_actual > 0 else 0
        raw_score += similarity * rarity_score

    # --- Normalization: Calculate the maximum possible score for this encounter ---
    # This is the score the 'actual' set would get against itself.
    max_possible_score = sum(idf_registry.get(term, 0.0) for term in actual_terms)

    if max_possible_score == 0:
        return 0.0

    # The final normalized score!
    return raw_score / max_possible_score


def evaluate_prediction_by_category(predicted: dict, actual: dict, idf_registry: dict, embedding_library: dict) -> dict[str, float]:
    """
    The main evaluation function. Replaces the old Jaccard score with our
    new Weighted Semantic Jujutsu. Now receives data directly instead of loading it.
    """
    scores = {}
    all_keys = ["diagnoses", "medications", "treatments"]
    
    for key in all_keys:
        pred_set = set(predicted.get(key, []))
        actual_set = set(actual.get(key, []))
        
        # This part works perfectly now, because it uses the data
        # we passed in as arguments!
        scores[key] = calculate_weighted_score(
            list(pred_set), 
            list(actual_set), 
            idf_registry, 
            embedding_library
        )

    # Calculate the overall score as a simple average of the category scores
    scores["overall"] = sum(scores.values()) / len(all_keys) if all_keys else 0.0
    return scores