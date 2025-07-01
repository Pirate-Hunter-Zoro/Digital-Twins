# --- main.py ---
import sys
import os
from functools import partial
import pickle

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scripts.parser import parse_data_args
import json
from multiprocessing import Pool
from scripts.read_data.load_patient_data import load_patient_data
from scripts.calculations.process_patient import process_patient
from scripts.llm.query_and_response import setup_prompt_generation
from scripts.config import setup_config, get_global_config

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

if __name__ == "__main__":
    args = parse_data_args()

    # --- FIX APPLIED: Connecting the new representation_method wire! ---
    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )
    # --- End of Fix ---
    
    global_config = get_global_config()

    print("--- Loading and Filtering Patient Data ---")
    patient_data_raw = load_patient_data()
    print(f"Loaded {len(patient_data_raw)} total patients.")

    required_visits = global_config.num_visits
    patient_data_filtered = [
        p for p in patient_data_raw if len(p.get("visits", [])) >= required_visits
    ]
    print(f"Filtered down to {len(patient_data_filtered)} patients with at least {required_visits} visits.")
    if len(patient_data_raw) > len(patient_data_filtered):
        print(f"Discarded {len(patient_data_raw) - len(patient_data_filtered)} patients due to insufficient visit history.")

    all_results = {}
    patients_processed = 0
    
    print("\n--- Loading Shared Data Libraries ---")
    try:
        with open('data/term_idf_registry.json', 'r') as f:
            idf_registry = json.load(f)
        with open("data/term_embedding_library_by_category.pkl", "rb") as f:
            embedding_library = pickle.load(f)
        print("...Shared data loaded and ready!")
    except FileNotFoundError as e:
        print(f"ERROR: Could not load required data libraries: {e}", file=sys.stderr)
        print("Please ensure you have run 'generate_idf_registry.py' and 'generate_term_embeddings.py' first.", file=sys.stderr)
        sys.exit(1)

    setup_prompt_generation()
    
    print(f"\n--- Starting Prediction Pool with {args.workers} Workers ---")
    process_pool = Pool(processes=args.workers)
    
    pool_results = process_pool.imap_unordered(process_patient, patient_data_filtered)
    
    # Filename now includes representation_method for perfect separation of experiment results!
    output_file = f"data/patient_results_{global_config.num_patients}_{global_config.num_visits}_{global_config.representation_method}_{global_config.vectorizer_method}_{global_config.distance_metric}.json"

    try:
        for patient_id, result in pool_results:
            all_results[patient_id] = convert_sets_to_lists(result)
            patients_processed += 1
            if patients_processed % args.save_every == 0:
                with open(output_file, "w") as f:
                    json.dump(all_results, f, indent=4)
                print(f"Saved results after processing {patients_processed} patients.")
    except Exception as e:
        print(f"An error occurred during multiprocessing: {e}", file=sys.stderr)
    finally:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nFinished processing {patients_processed} patients. Final results saved to {output_file}.")

        process_pool.close()
        process_pool.join()
        print("--- All processes complete! ---")