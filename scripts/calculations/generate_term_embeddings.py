import os
import sys
import json
import argparse
import pickle
from sentence_transformers import SentenceTransformer

# --- Dynamic Pathing ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
sys.path.insert(0, project_root)
from scripts.config import setup_config, get_global_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", required=True)
    parser.add_argument("--num_visits", type=int, required=True)
    parser.add_argument("--num_patients", type=int, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
    )

    config = get_global_config()

    model_dir = os.path.join(project_root, "models", config.vectorizer_method)
    embedder = SentenceTransformer(model_dir, local_files_only=True)

    input_path = os.path.join(project_root, "data", f"grouped_terms_by_category_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.json")
    output_path = os.path.join(
        project_root,
        "data",
        f"term_embedding_library_by_category_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.pkl"
    )

    print(f"📥 Loading input terms from: {input_path}")
    with open(input_path, "r") as f:
        terms_by_category = json.load(f)

    term_library = {}
    for category, terms in terms_by_category.items():
        vectors = embedder.encode(terms, convert_to_numpy=True)
        term_library[category] = [
            {"term": term, "embedding": vector.tolist()}
            for term, vector in zip(terms, vectors)
        ]

    with open(output_path, "wb") as f:
        pickle.dump(term_library, f)

    print(f"✅ Saved embedded terms to: {output_path}")

if __name__ == "__main__":
    main()
