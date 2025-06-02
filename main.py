import argparse
import json
from multiprocessing import Pool
from generate_patients import load_patient_data
from process_patient import process_patient
from query_llm import query_llm

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
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes to use.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of saving results per patients processed.")
    parser.add_argument("--output", type=str, default="patient_output_results.json", help="Output directory to save results.")
    args = parser.parse_args()

    patient_data = load_patient_data()

    all_results = {}

    patients_processed = 0

    process_pool = Pool(processes=args.workers)
    pool_results = process_pool.imap_unordered(process_patient, patient_data)
    # Collect and save results from the pool as they become available
    for patient_id, result in pool_results:
        all_results[patient_id] = convert_sets_to_lists(result)
        patients_processed += 1
        if patients_processed % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=4)
            print(f"Saved results after processing {patients_processed} patients.")

    # Final save after all processing is done
    with open("output.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Finished processing {patients_processed} patients. Results saved to {args.output}.")

    process_pool.close()
    process_pool.join()