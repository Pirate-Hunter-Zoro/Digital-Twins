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

def evaluate_prediction_by_category(predicted, actual, term_embeddings, similarity_threshold=0.4):
    categories = ["diagnoses", "medications", "treatments"]
    results = {}
    
    for category in categories:
        pred_terms = predicted.get(category, [])
        actual_terms = actual.get(category, [])
        
        pred_vecs = [term_embeddings[clean_term(t)] for t in pred_terms if clean_term(t) in term_embeddings]
        actual_vecs = [term_embeddings[clean_term(t)] for t in actual_terms if clean_term(t) in term_embeddings]
        
        if not pred_vecs or not actual_vecs:
            results[category] = 0.0
            continue

        sim_matrix = cosine_similarity_matrix(np.array(pred_vecs), np.array(actual_vecs))
        matches = (sim_matrix.max(axis=1) >= similarity_threshold).sum()
        score = matches / len(pred_terms)
        results[category] = score

    results["overall"] = np.mean(list(results.values()))
    return results
