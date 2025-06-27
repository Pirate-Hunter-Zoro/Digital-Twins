# --- parser.py ---
import sys
import os
from argparse import ArgumentParser

# --- Dynamic sys.path adjustment ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path adjustment ---

def parse_data_args():
    parser = ArgumentParser(description="Process patient data and generate results.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of saving results per patients processed.")
    
    # --- NEW UPGRADE SLOT! ---
    parser.add_argument("--representation_method", type=str, default="visit_sentence", choices=["visit_sentence", "bag_of_codes"], help="Method for representing patient history.")

    parser.add_argument("--vectorizer_method", type=str, default="biobert-mnli-medlini", choices=["biobert-mnli-medlini", "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"], help="Method for vectorization.")
    parser.add_argument("--distance_metric", type=str, default="euclidean", choices=["cosine", "euclidean"], help="Distance metric for nearest neighbors.")
    parser.add_argument("--num_visits", type=int, default=6, help="Number of visits to consider for each patient history.")
    parser.add_argument("--num_patients", type=int, default=5000, help="Number of patients to process (random subset of the population).")
    parser.add_argument("--num_neighbors", type=int, default=10, help="Number of nearest neighbors to consider.")
    return parser.parse_args()