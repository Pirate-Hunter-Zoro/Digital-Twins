import json
import pickle
from generate_patients import load_patient_data
from llm_helper import get_narrative, get_relevance_score
from scipy.spatial.distance import mahalanobis
import numpy as np
from config import setup_config, get_global_config
import argparse

def inspect_visit(patient_id: str,k: int = 5) -> None:
    output = []
    output_path = f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/neighbor_inspection_{patient_id}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.txt"

    patients = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patients}

    with open(f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/patient_results_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json", "r") as f:
        patient_output_results = json.load(f)

    with open(f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/all_prompts_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json", "r") as f:
        all_prompts = json.load(f)

    with open(f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl", "rb") as f:
        nearest_neighbors = pickle.load(f)

    with open(f"{'synthetic_data' if get_global_config().use_synthetic_data else 'real_data'}/all_vectors_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl", "rb") as f:
        all_vectors = pickle.load(f)

    key = (patient_id, get_global_config().num_visits - 1)  # We use the latest visit *up to* the prediction point
    neighbors = nearest_neighbors.get(key)
    if not neighbors:
        output.append(f"No neighbors found for {key}\n")
    else:
        n = len(neighbors)
        if 2 * k > n:
            output.append(f"Not enough neighbors (have {n}, need {2*k})\n")
        else:
            prompt_key = f"{patient_id}_{get_global_config().num_visits - 1}"
            prompt = all_prompts.get(prompt_key)
            target_narrative = get_narrative(patient_lookup[patient_id]["visits"], get_global_config().num_visits - 1)
            output.append(f"\nTarget Patient Narrative (up to Visit {get_global_config().num_visits-1}):\n{target_narrative}\n")
            if prompt is None:
                output.append(f"No prompt found for {prompt_key}\n")
            else:
                output.append("Closest Sequence of Visits from Other Patients (later is closer):")
                for i in range(k - 1, -1, -1):
                    (neighbor_pid, neighbor_vidx), similarity, _ = neighbors[i]
                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"], neighbor_vidx)
                    output.append(f"  {k - i}. ID: ({neighbor_pid}, {neighbor_vidx}), Cosine Similarity: {similarity:.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")
                    relevance_score = get_relevance_score(target_narrative, narrative)
                    output.append(f"    Relevance Score: {relevance_score:.4f}\n")

                output.append("Farthest Sequence of Visits from Other Patients (later is farther):")
                for i in range(n - k, n):
                    (neighbor_pid, neighbor_vidx), similarity, _ = neighbors[i]
                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"], neighbor_vidx)
                    output.append(f"  {i - (n - k) + 1}. ID: ({neighbor_pid}, {neighbor_vidx}), Distance: {similarity:.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")
                    relevance_score = get_relevance_score(target_narrative, narrative)
                    output.append(f"    Relevance Score: {relevance_score:.4f}\n")

                result = patient_output_results[patient_id][get_global_config().num_visits - 1]
                output.append(f"Predicted: {result['predicted']}")
                output.append(f"Actual: {result['actual']}")
                output.append(f"Scores: {result.get('scores', {})}")
               
                # Compute the Mahalanobis distance for the target visit from all the nearest neighbors
                nearest_neighbor_vectors = [neighbor[2] for neighbor in neighbors[:k]]
                target_vector = all_vectors.get(key)

                if target_vector is not None and len(nearest_neighbor_vectors)>1:
                    try:
                        covariance_matrix = np.cov(np.array(nearest_neighbor_vectors).T)
                        inv_cov_matrix = np.linalg.inv(covariance_matrix)
                        distance = mahalanobis(target_vector, np.mean(nearest_neighbor_vectors, axis=0), inv_cov_matrix)
                        output.append(f"Mahalanobis distance from target visit to nearest neighbors: {distance:.4f}\n")
                    except np.linalg.LinAlgError:
                        output.append("Covariance matrix is singular, cannot compute Mahalanobis distance.\n")

    with open(output_path, "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectorizer_method", type=str, default="sentence_transformer", help="Method for vectorization (e.g., 'sentence_transformer', 'tfidf').")
    parser.add_argument("--distance_metric", type=str, default="cosine", help="Distance metric to use for nearest neighbors (e.g., 'cosine', 'euclidean').")
    parser.add_argument("--use_synthetic_data", type=bool, default=False, help="Use synthetic data for testing purposes.")
    parser.add_argument("--num_visits", type=int, default=5, help="Number of visits to consider for each patient.")
    parser.add_argument("--num_patients", type=int, default=100, help="Number of patients to process (random subset of the real or synthetic population).")
    args = parser.parse_args()

    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        use_synthetic_data=args.use_synthetic_data,
        num_visits=args.num_visits,
        num_patients=args.num_patients
    )

    inspect_visit(patient_id = "P0000001")
