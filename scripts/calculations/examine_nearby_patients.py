import sys
import os

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

# Add the project root to sys.path if it's not already there.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import json
import pickle
from scripts.read_data.load_patient_data import load_patient_data
from scripts.llm.llm_helper import get_narrative, get_relevance_score
from scipy.spatial.distance import mahalanobis
import numpy as np
from scripts.config import setup_config, get_global_config
from scripts.parser import parse_data_args

# --- NEW: Global list to collect correlation data ---
# This will collect results for all patients inspected when the script is run.
correlation_data_collector = []

def inspect_visit(patient_id: str, k: int = 5) -> None:
    output = [] # This is for the human-readable report
    output_path = f"real_data/neighbor_inspection_{patient_id}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.txt"

    patients = load_patient_data()
    patient_lookup = {p["patient_id"]: p for p in patients}

    # Load previously generated results
    patient_results_file = f"real_data/patient_results_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json"
    if not os.path.exists(patient_results_file):
        output.append(f"Error: Patient results file not found at {patient_results_file}. Cannot inspect.\n")
        with open(output_path, "w") as f: f.write("\n".join(output))
        return # Exit if results file is missing
    
    with open(patient_results_file, "r") as f:
        patient_output_results = json.load(f)

    # Load all_prompts (optional, if you want to inspect them)
    all_prompts_file = f"real_data/all_prompts_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json"
    if not os.path.exists(all_prompts_file):
        output.append(f"Warning: All prompts file not found at {all_prompts_file}.\n")
        all_prompts = {}
    else:
        with open(all_prompts_file, "r") as f:
            all_prompts = json.load(f)


    # Load nearest neighbors (these should be from the pre-computed pipeline run)
    neighbors_file = f"real_data/neighbors_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.pkl"
    if not os.path.exists(neighbors_file):
        output.append(f"Error: Nearest neighbors file not found at {neighbors_file}. Cannot inspect.\n")
        with open(output_path, "w") as f: f.write("\n".join(output))
        return # Exit if neighbors file is missing

    with open(neighbors_file, "rb") as f:
        nearest_neighbors = pickle.load(f)

    # Load all_vectors (from compute_nearest_neighbors.py)
    all_vectors_file = f"real_data/all_vectors_{get_global_config().vectorizer_method}_{get_global_config().num_visits}.pkl"
    if not os.path.exists(all_vectors_file):
        output.append(f"Error: All vectors file not found at {all_vectors_file}. Cannot inspect.\n")
        with open(output_path, "w") as f: f.write("\n".join(output))
        return # Exit if all_vectors file is missing

    with open(all_vectors_file, "rb") as f:
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
            target_narrative = get_narrative(patient_lookup[patient_id]["visits"][:get_global_config().num_visits]) # Ensure history is same length as prompt

            output.append(f"\nTarget Patient Narrative (up to Visit {get_global_config().num_visits-1}):\n{target_narrative}\n")
            if prompt is None:
                output.append(f"No prompt found for {prompt_key}\n")
            else:
                # --- Collect LLM Relevance Scores for Average ---
                llm_relevance_scores_for_avg = []
                
                output.append("Closest Sequence of Visits from Other Patients (later is closer):")
                for i in range(k - 1, -1, -1): # Iterate from closest to slightly less close (k closest)
                    (neighbor_pid, neighbor_vidx), similarity, neighbor_vector = neighbors[i]
                    
                    # Ensure neighbor_pid and neighbor_vidx are valid keys
                    neighbor_key = (neighbor_pid, neighbor_vidx)
                    if neighbor_key not in all_vectors or neighbor_pid not in patient_lookup:
                        output.append(f"  Warning: Neighbor {neighbor_key} not found in all_vectors or patient_lookup. Skipping.\n")
                        continue

                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"][:neighbor_vidx + 1]) # Ensure history matches neighbor_vidx
                    output.append(f"  {k - i}. ID: ({neighbor_pid}, {neighbor_vidx}), Distance: {abs(similarity):.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")
                    
                    relevance_score = get_relevance_score(target_narrative, narrative)
                    llm_relevance_scores_for_avg.append(relevance_score) # Collect score for average
                    output.append(f"    Relevance Score: {relevance_score:.4f}\n")

                # Calculate average LLM relevance score
                avg_llm_relevance_score = np.mean(llm_relevance_scores_for_avg) if llm_relevance_scores_for_avg else 0.0
                output.append(f"Average LLM Relevance Score for {k} closest neighbors: {avg_llm_relevance_score:.4f}\n")

                # Farthest neighbors (same logic)
                output.append("Farthest Sequence of Visits from Other Patients (later is farther):")
                for i in range(n - k, n):
                    (neighbor_pid, neighbor_vidx), similarity, neighbor_vector = neighbors[i]
                    
                    # Ensure neighbor_pid and neighbor_vidx are valid keys
                    neighbor_key = (neighbor_pid, neighbor_vidx)
                    if neighbor_key not in all_vectors or neighbor_pid not in patient_lookup:
                        output.append(f"  Warning: Neighbor {neighbor_key} not found in all_vectors or patient_lookup. Skipping.\n")
                        continue

                    narrative = get_narrative(patient_lookup[neighbor_pid]["visits"][:neighbor_vidx + 1])
                    output.append(f"  {i - (n - k) + 1}. ID: ({neighbor_pid}, {neighbor_vidx}), Distance: {abs(similarity):.4f}")
                    output.append(f"    Patient Narrative: {narrative}\n")
                    relevance_score = get_relevance_score(target_narrative, narrative)
                    # Don't add to avg list here, as we only average closest
                    output.append(f"    Relevance Score: {relevance_score:.4f}\n")


                result = patient_output_results.get(patient_id, {}) # Use .get for safety
                if result:
                    output.append(f"Predicted: {result.get('predicted', {})}")
                    output.append(f"Actual: {result.get('actual', {})}")
                    output.append(f"Scores: {result.get('scores', {})}")
                else:
                    output.append(f"Warning: No prediction results found for patient {patient_id}.\n")
               
                # Compute the Mahalanobis distance for the target visit from all the nearest neighbors
                nearest_neighbor_vectors = [neighbor[2] for neighbor in neighbors[:k]]
                target_vector = all_vectors.get(key)
                
                mahalanobis_distance = 0.0 # Initialize Mahalanobis distance
                if target_vector is not None and len(nearest_neighbor_vectors) > 1:
                    try:
                        # Ensure all vectors are np.ndarray type and have consistent dimensions
                        nearest_neighbor_vectors_np = np.array(nearest_neighbor_vectors)
                        target_vector_np = np.array(target_vector)

                        # Compute covariance matrix only if there's enough data (more than 1 sample)
                        # np.cov needs at least 2 samples if data is 1D array.
                        # If the vectors are 2D arrays (multiple features), it needs more.
                        if nearest_neighbor_vectors_np.shape[0] > 1:
                            # If covariance_matrix is singular, it implies perfect linear correlation
                            # or insufficient number of samples vs dimensions.
                            # Add a small regularization term (e.g., identity matrix scaled by epsilon)
                            # to the covariance matrix if it's singular.
                            covariance_matrix = np.cov(nearest_neighbor_vectors_np.T)
                            
                            # Add a small diagonal to avoid singularity if needed for inverse
                            epsilon = 1e-6 * np.eye(covariance_matrix.shape[0])
                            inv_cov_matrix = np.linalg.inv(covariance_matrix + epsilon)
                            
                            mahalanobis_distance = mahalanobis(target_vector_np, np.mean(nearest_neighbor_vectors_np, axis=0), inv_cov_matrix)
                            output.append(f"Mahalanobis distance from target visit to {k} nearest neighbors: {mahalanobis_distance:.4f}\n")
                        else:
                            output.append(f"Not enough nearest neighbor vectors ({len(nearest_neighbor_vectors)}) to compute Mahalanobis distance.\n")
                    except np.linalg.LinAlgError:
                        output.append("Covariance matrix is singular, cannot compute Mahalanobis distance (after regularization).\n")
                    except Exception as e:
                        output.append(f"Error computing Mahalanobis distance: {e}\n")

                # --- NEW: Collect data for correlation ---
                # Only collect if both measures are valid
                if avg_llm_relevance_score is not None and mahalanobis_distance is not None:
                    correlation_data_collector.append({
                        "patient_id": patient_id,
                        "visit_idx": get_global_config().num_visits - 1, # The index of the predicted visit
                        "avg_llm_relevance_score": avg_llm_relevance_score,
                        "mahalanobis_distance": mahalanobis_distance
                    })


    with open(output_path, "w") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    args = parse_data_args()

    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        use_synthetic_data=False, # Ensure real data path for inspection
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    
    # --- Collect patients to inspect ---
    # The current inspect_visit only takes one patient_id.
    # We need to loop through all patients to collect correlation data.

    # Load all patients data (using the new load_patient_data.py)
    all_patients_for_inspection = load_patient_data()
    
    # Filter patients if num_patients is set in config (from process_data.py context)
    # This ensures inspect_visit is called for the same subset as main.py would process.
    if get_global_config().num_patients < len(all_patients_for_inspection):
        # Sample for inspection based on the same random_state as process_data.py
        all_patients_for_inspection = pd.DataFrame(all_patients_for_inspection).sample(n=get_global_config().num_patients, random_state=42).to_dict(orient='records')
        print(f"Inspecting a sample of {len(all_patients_for_inspection)} patients.")

    # Only inspect patients who have enough visits for the configured num_visits history
    eligible_patients = [
        p for p in all_patients_for_inspection 
        if len(p.get("visits", [])) >= get_global_config().num_visits
    ]
    print(f"Inspecting {len(eligible_patients)} eligible patients out of {len(all_patients_for_inspection)} total sampled.")

    if not eligible_patients:
        print("No eligible patients found for inspection. Cannot generate correlation data.")
    else:
        # Loop through each eligible patient and run inspect_visit
        for patient_to_inspect in eligible_patients:
            print(f"Inspecting patient: {patient_to_inspect['patient_id']}...")
            inspect_visit(patient_id=patient_to_inspect['patient_id'])

        # --- Save the collected correlation data to a JSON file ---
        correlation_output_file = f"real_data/llm_mahalanobis_correlation_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json"
        
        try:
            with open(correlation_output_file, "w") as f:
                json.dump(correlation_data_collector, f, indent=4)
            print(f"\nCorrelation data saved to: {correlation_output_file}")
        except Exception as e:
            print(f"Error saving correlation data: {e}", file=sys.stderr)

    print("Inspection completed.")