import numpy as np
import re

def clean_term(term: str) -> str:
    term = term.lower().strip()
    term = re.sub(r"\s*\([^)]*hcc[^)]*\)", "", term)
    term = re.sub(r"\b\d{3}\.\d{1,2}\b", "", term)
    blacklist = ["initial encounter", "unspecified", "nos", "nec", "<none>", "<None>", ";", ":"]
    for noise in blacklist:
        term = term.replace(noise, "")
    term = re.sub(r"\s+", " ", term)
    return term.strip()

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if b.ndim == 1:
        b = b[np.newaxis, :]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)

def evaluate_prediction_by_category(predicted, actual, idf_registry, embedding_library):
    categories = ["diagnoses", "medications", "treatments"]
    results = {}
    matched_pairs_by_category = {}

    for category in categories:
        pred_terms = predicted.get(category, [])
        actual_terms = actual.get(category, [])

        # Clean and filter for terms with embeddings
        pred_terms_clean = [t for t in map(clean_term, pred_terms) if t in embedding_library]
        actual_terms_clean = [t for t in map(clean_term, actual_terms) if t in embedding_library]

        if not pred_terms_clean or not actual_terms_clean:
            results[category] = 0.0
            matched_pairs_by_category[category] = []
            continue

        pred_vecs = np.vstack([embedding_library[t] for t in pred_terms_clean])
        actual_vecs = np.vstack([embedding_library[t] for t in actual_terms_clean])

        similarity_matrix = cosine_similarity_matrix(actual_vecs, pred_vecs)

        actual_with_idf = sorted(
            [(i, t, idf_registry.get(t, 1.0)) for i, t in enumerate(actual_terms_clean)],
            key=lambda x: -x[2]
        )

        used_pred_indices = set()
        score = 0.0
        matched_pairs = []

        for actual_idx, actual_term, actual_idf in actual_with_idf:
            best_score = 0.0
            best_pred_idx = None
            best_pred_term = None

            for pred_idx, pred_term in enumerate(pred_terms_clean):
                if pred_idx in used_pred_indices:
                    continue
                sim = max(0.0, similarity_matrix[actual_idx, pred_idx])
                pred_idf = idf_registry.get(pred_term, 1.0)
                weighted_score = sim * np.sqrt(actual_idf * pred_idf)

                if weighted_score > best_score:
                    best_score = weighted_score
                    best_pred_idx = pred_idx
                    best_pred_term = pred_term

            if best_pred_idx is not None:
                used_pred_indices.add(best_pred_idx)
                score += best_score
                matched_pairs.append({
                    "actual_term": actual_term,
                    "predicted_term": best_pred_term,
                    "similarity": float(similarity_matrix[actual_idx, best_pred_idx]),
                    "actual_idf": actual_idf,
                    "predicted_idf": idf_registry.get(best_pred_term, 1.0),
                    "weighted_score": best_score,
                })

        results[category] = score
        matched_pairs_by_category[category] = matched_pairs

    scores = [v for v in results.values() if isinstance(v, (float, int))]
    results["overall"] = np.mean(scores) if scores else 0.0
    matched_pairs_by_category["overall_score"] = results["overall"]

    return results, matched_pairs_by_category
