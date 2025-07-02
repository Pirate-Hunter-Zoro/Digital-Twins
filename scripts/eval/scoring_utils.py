import numpy as np

def compute_weighted_cosine_score(similarity_matrix, actual_terms, predicted_terms, idf_registry):
    """
    Computes the weighted cosine similarity score between actual and predicted terms.
    Score = cosine_similarity × sqrt(actual_idf * predicted_idf)

    Args:
        similarity_matrix (np.ndarray): Cosine similarities between all actual and predicted terms.
        actual_terms (list[str]): The actual terms from the patient data.
        predicted_terms (list[str]): The predicted terms from the LLM.
        idf_registry (dict[str, float]): Mapping of term to inverse document frequency.

    Returns:
        float: Final semantic similarity score.
    """
    score = 0.0
    matched = set()
    for actual_idx, actual_term in enumerate(actual_terms):
        best_score = 0.0
        for pred_idx, pred_term in enumerate(predicted_terms):
            if pred_idx in matched:
                continue
            sim = max(0.0, similarity_matrix[actual_idx, pred_idx])
            actual_idf = idf_registry.get(actual_term, 1.0)
            pred_idf = idf_registry.get(pred_term, 1.0)
            weighted_score = sim * np.sqrt(actual_idf * pred_idf)
            if weighted_score > best_score:
                best_score = weighted_score
                best_pred_idx = pred_idx
        if best_score > 0.0:
            matched.add(best_pred_idx)
            score += best_score
    return score
