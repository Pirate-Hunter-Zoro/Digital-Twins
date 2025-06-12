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
    parser.add_argument("--use_synthetic_data", type=bool, default=True, help="Use synthetic data for testing purposes.")
    parser.add_argument("--num_visits", type=int, default=5, help="Number of visits to consider for each patient.")
    parser.add_argument("--num_patients", type=int, default=50, help="Number of patients to process (random subset of the real or synthetic population).")
    parser.add_argument("--num_neighbors", type=int, default=5, help="Number of nearest neighbors to consider for each visit.")
    return parser.parse_args()

