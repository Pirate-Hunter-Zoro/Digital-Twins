import json
import pickle
from generate_patients import load_patient_data
from llm_helper import get_narrative, get_relevance_score
from scipy.spatial.distance import mahalanobis
import numpy as np

def inspect_visit(patient_id: str, visit_idx: int, k: int = 5, vectorizor: str="sentence_transformer", distance_metric: str="cosine") -> None:
    output = []
    output_path = f"neighbor_inspection_{patient_id}_{visit_idx}_{vectorizor}_{distance_metric}.txt"

    patients = load_patient_data(vectorizor=vectorizor, distance_metric=distance_metric)
    patient_lookup = {p["patient_id"]: p for p in patients}

    with open(f"patient_lookup_{vectorizor}_{distance_metric}.json", "r") as f:
        patient_output_results = json.load(f)

    with open(f"all_prompts_{vectorizor}_{distance_metric}.json", "r") as f:
        all_prompts = json.load(f)

    with open(f"nearest_neighbors_{vectorizor}_{distance_metric}.pkl", "rb") as f:
        nearest_neighbors = pickle.load(f)

    with open(f"all_vectors_{vectorizor}_{visit_idx}.pkl", "rb") as f:
        all_vectors = pickle.load(f)

    key = (patient_id, visit_idx)
    neighbors = nearest_neighbors.get(key)
    if not neighbors:
        output.append(f"No neighbors found for {key}\n")
    else:
        n = len(neighbors)
        if 2 * k > n:
            output.append(f"Not enough neighbors (have {n}, need {2*k})\n")
        else:
            prompt_key = f"{patient_id}_{visit_idx}"
            prompt = all_prompts.get(prompt_key)
            target_narrative = get_narrative(patient_lookup[patient_id]["visits"], visit_idx)
            output.append(f"\nTarget Patient Narrative (up to Visit {visit_idx}):\n{target_narrative}\n")
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
                    output.append(f"  {i - (n - k) + 1}. ID: ({neighbor_pid}, {neighbor_vidx}), Cosine Similarity: {similarity:.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")
                    relevance_score = get_relevance_score(target_narrative, narrative)
                    output.append(f"    Relevance Score: {relevance_score:.4f}\n")

                results = patient_output_results.get(patient_id, [])
                if visit_idx < len(results):
                    result = results[visit_idx]
                    output.append(f"Predicted: {result['predicted']}")
                    output.append(f"Actual: {result['actual']}")
                    output.append(f"Scores: {result.get('scores', {})}")
                else:
                    output.append("No prediction result found for this visit.")

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
    inspect_visit("P0000001", 3, 2)  # Output will be written to 'examine_output.txt' by default
