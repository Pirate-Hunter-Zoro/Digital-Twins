def evaluate_prediction_by_category(predicted: dict, actual: dict) -> dict[str, float]:
    scores = {}
    all_keys = ["diagnoses", "medications", "treatments"]
    for key in all_keys:
        pred_set = predicted.get(key, set())
        actual_set = actual.get(key, set())
        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)
        scores[key] = intersection / union if union else 0.0
    scores["overall"] = sum(scores.values()) / len(all_keys)
    return scores