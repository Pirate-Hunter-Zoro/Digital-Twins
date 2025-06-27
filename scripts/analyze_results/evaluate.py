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
    
    for category in categories:
        pred_terms = predicted.get(category, [])
        actual_terms = actual.get(category, [])
        
        pred_vecs = [embedding_library[clean_term(t)] for t in pred_terms if clean_term(t) in embedding_library]
        actual_vecs = [embedding_library[clean_term(t)] for t in actual_terms if clean_term(t) in embedding_library]
        
        score = 0.0
        
        
        
        results[category] = score

    results["overall"] = np.mean(list(results.values()))
    return results
