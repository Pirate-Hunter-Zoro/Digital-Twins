import sys
import os
from functools import partial
import pickle

# --- Dynamic sys.path adjustment for module imports ---
# Get the absolute path to the directory containing the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the project root.
# This assumes your 'config.py' (or other root-level modules like 'main.py')
# is located two directories up from where this script resides.
# Example: If this script is in 'project_root/scripts/read_data/your_script.py'
# then '..' takes you to 'project_root/scripts/'
# and '..', '..' takes you to 'project_root/'
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at index 0 makes it the first place Python looks for modules,
# so 'import config' will find it quickly.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- End of sys.path adjustment ---

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from parser import parse_data_args
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

    setup_config(
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
    )

    patient_data = load_patient_data()
    all_results = {}
    patients_processed = 0
    
    # ==========================================================
    # --- ENTRAPTA'S UPGRADE, PART 1: LOAD SHARED DATA HERE ---
    # ==========================================================
    # Load the big data files ONCE, right here.
    print("Loading shared data libraries once...")
    # NOTE: You might need to adjust the file paths if they aren't in the root!
    # The README mentions a `term_idf_registry.json` and `term_embedding_library.pkl`
    with open('term_idf_registry.json', 'r') as f:
        idf_registry = json.load(f)
    with open('term_embedding_library.pkl', 'rb') as f:
        embedding_library = pickle.load(f)
    print("...Shared data loaded and ready!")
    # ==========================================================

    setup_prompt_generation()

    # Use `partial` to create a new version of your worker function
    # that has the shared data "baked in" as arguments.
    worker_with_data = partial(process_patient, 
                               idf_registry=idf_registry, 
                               embedding_library=embedding_library)
    # ===================================================================

    process_pool = Pool(processes=args.workers)
    
    pool_results = process_pool.imap_unordered(worker_with_data, patient_data)
    # =====================================================================

    output_file = f"real_data/patient_results_{get_global_config().num_patients}_{get_global_config().num_visits}_{get_global_config().vectorizer_method}_{get_global_config().distance_metric}.json"

    try:
        for patient_id, result in pool_results:
            all_results[patient_id] = convert_sets_to_lists(result)
            patients_processed += 1
            if patients_processed % args.save_every == 0:
                with open(output_file, "w") as f:
                    json.dump(all_results, f, indent=4)
                print(f"Saved results after processing {patients_processed} patients.")
    except Exception as e:
        print(f"Error during multiprocessing: {e}")

    # Final save
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Finished processing {patients_processed} patients. Results saved to {output_file}.")

    process_pool.close()
    process_pool.join()