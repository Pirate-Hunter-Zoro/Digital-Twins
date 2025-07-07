import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

current_script_dir = Path(__file__).resolve().parent
scripts_dir = current_script_dir.parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from calculations.prepare_categorized_embedding_terms import clean_term
from config import setup_config, get_global_config

def get_sentence_embedder(model_path):
    try:
        return SentenceTransformer(str(model_path), local_files_only=True), "sentence-transformers"
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)

        def embed_fn(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            return cls_embedding[0].numpy()
        return embed_fn, "hf-transformers"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation_method", required=True)
    parser.add_argument("--vectorizer_method", required=True)
    parser.add_argument("--distance_metric", default="euclidean")
    parser.add_argument("--num_visits", type=int, default=6)
    parser.add_argument("--num_patients", type=int, default=5000)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--model_name", default="medgemma")
    args = parser.parse_args()

    setup_config(
        representation_method=args.representation_method,
        vectorizer_method=args.vectorizer_method,
        distance_metric=args.distance_metric,
        num_visits=args.num_visits,
        num_patients=args.num_patients,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name
    )
    config = get_global_config()

    current_script_dir = Path(__file__).resolve().parent
    project_root = current_script_dir.parents[2]
    project_loc = project_root.parent

    with open(project_root / "data" / "grouped_terms_by_category.json", "r") as f:
        term_data = json.load(f)

    model_path = project_loc / "models" / config.vectorizer_method
    embedder, embed_type = get_sentence_embedder(model_path)

    term_embedding_library = {}

    for category, terms in term_data.items():
        term_embedding_library[category] = []
        for term in terms:
            cleaned = clean_term(term)
            embedding = embedder.encode(cleaned) if embed_type == "sentence-transformers" else embedder(cleaned)
            term_embedding_library[category].append({
                "term": term,
                "cleaned": cleaned,
                "embedding": embedding.tolist()
            })

    out_path = project_root / "data" / f"term_embedding_library_by_category_{config.num_patients}_{config.num_visits}_{config.representation_method}_{config.vectorizer_method}_{config.distance_metric}_{config.model_name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(term_embedding_library, f)

    print(f"✅ Saved term embedding library to {out_path}")

if __name__ == "__main__":
    main()
