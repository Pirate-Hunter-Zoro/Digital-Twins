import sys
import os

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

from argparse import ArgumentParser

def parse_data_args() -> list[str]:
    """Returns the parsed command line arguments.
    This function sets up the argument parser with various options for processing patient data,
    including the number of workers, saving frequency, vectorization method, distance metric,
    whether to use synthetic data, number of visits, number of patients, and number of neighbors.
    It returns the parsed arguments as a list of strings.

    Returns:
        list[str]: Parsed command line arguments.
    """
    parser = ArgumentParser(description="Process patient data and generate results.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes to use.")
    parser.add_argument("--save_every", type=int, default=10, help="Frequency of saving results per patients processed.")
    parser.add_argument("--vectorizer_method", type=str, default="sentence_transformer", help="Method for vectorization (e.g., 'sentence_transformer', 'tfidf').")
    parser.add_argument("--distance_metric", type=str, default="euclidean", help="Distance metric to use for nearest neighbors (e.g., 'cosine', 'euclidean').")
    parser.add_argument("--num_visits", type=int, default=5, help="Number of visits to consider for each patient.")
    parser.add_argument("--num_patients", type=int, default=50, help="Number of patients to process (random subset of the real or synthetic population).")
    parser.add_argument("--num_neighbors", type=int, default=5, help="Number of nearest neighbors to consider for each visit.")
    return parser.parse_args()