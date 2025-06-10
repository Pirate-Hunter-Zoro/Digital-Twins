import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
from multiprocessing import Pool
from generate_patients import load_patient_data
from process_patient import process_patient
from query_llm import query_llm
from query_and_response import setup

def convert_sets_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(v) for v in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

querying = False

if __name__ == "__main__":
    if querying:
        query = "What are the diagnoses, medications, and treatments for a patient with symptoms of diabetes and hypertension? Return as JSON."
        print("Querying LLM with the following question:")
        print(query)
        print(query_llm(query))
        print("LLM query complete. Now processing patient data...")

    parser = argparse.ArgumentParser(description="Evaluate all JSON files in a directory in parallel.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of saving results per patients processed.")
    parser.add_argument("--vectorizer_method", type=str, default="sentence_transformer", help="Method for vectorization (e.g., 'sentence_transformer', 'tfidf').")
    parser.add_argument("--distance_metric", type=str, default="cosine", help="Distance metric to use for nearest neighbors (e.g., 'cosine', 'euclidean').")
    parser.add_argument("--use_synthetic_data", type=bool, default=False, help="Use synthetic data for testing purposes.")
    parser.add_argument("--num_visits", type=int, default=5, help="Number of visits to consider for each patient.")
    parser.add_argument("--num_patients", type=int, default=100, help="Number of patients to process (random subset of the real or synthetic population).")
    args = parser.parse_args()

    patient_data = load_patient_data(use_synthetic_data=args.use_synthetic_data, num_visits=args.num_visits, num_patients=args.num_patients)
    all_results = {}
    patients_processed = 0

    setup(use_synthetic_data=args.use_synthetic_data, num_patients=args.num_patients, num_visits=args.num_visits)

    # We need a partial function to run process_patient with the additional parameters from args
    from functools import partial
    partial_process_patient = partial(
        process_patient,
        vectorizer=args.vectorizer_method,
        distance_metric=args.distance_metric
    )

    process_pool = Pool(processes=args.workers)
    pool_results = process_pool.imap_unordered(partial_process_patient, patient_data)
    output_file = f"patient_results_{args.vectorizer_method}_{args.distance_metric}"

    try:
        for patient_id, result in pool_results:
            all_results[patient_id] = convert_sets_to_lists(result)
            patients_processed += 1
            if patients_processed % args.save_every == 0:
                with open(args.output, "w") as f:
                    json.dump(all_results, f, indent=4)
                print(f"Saved results after processing {patients_processed} patients.")
    except Exception as e:
        print(f"Error during multiprocessing: {e}")

    # Final save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Finished processing {patients_processed} patients. Results saved to {args.output}.")

    process_pool.close()
    process_pool.join()