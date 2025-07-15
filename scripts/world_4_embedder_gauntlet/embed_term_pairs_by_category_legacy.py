import os
import sys
import pandas as pd
import argparse
from pathlib import Path
from gensim.models import KeyedVectors

# --- Dynamic sys.path adjustment ---
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.common.config import setup_config, get_global_config

def load_legacy_model(model_path, is_word2vec):
    """Loads a GloVe or Word2Vec model."""
    print(f"-> Loading legacy model from {model_path}...")
    binary = is_word2vec
    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
        print("-> Model loaded!")
        return model
    except Exception as e:
        print(f"❌ ERROR: Failed to load model {model_path}. Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Embed term pairs for LEGACY models.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--is_word2vec", action="store_true", help="Flag if the model is in Word2Vec binary format.")
    args = parser.parse_args()

    # --- ✨ NEW: Pre-computation Check ✨ ---
    output_check_path = project_root / "data" / "embeddings_by_category" / "procedure" / f"{args.model_name}.csv"
    if os.path.exists(output_check_path):
        print(f"✅ Hooray! Legacy results for {args.model_name} already exist at {output_check_path}. Skipping!")
        return

    print(f"🚀 Starting legacy embedding process for model: {args.model_name}")

    setup_config(vectorizer_method=args.model_name)
    config = get_global_config()

    # --- Path setup ---
    data_dir = project_root / "data"
    pairs_path = data_dir / "term_pairs_by_category.json"
    output_base_dir = data_dir / "embeddings_by_category"

    # --- Load Model ---
    model = load_legacy_model(args.model_path, args.is_word2vec)
    if model is None:
        sys.exit(1) # Exit if the model fails to load

    # --- Load Term Pairs ---
    print(f"📂 Loading term pairs from: {pairs_path}")
    with open(pairs_path, 'r') as f:
        term_pairs_by_category = json.load(f)

    # --- Process each category ---
    for category, pairs in term_pairs_by_category.items():
        print(f"\n--- Processing category: {category} ---")
        if not pairs:
            print("No pairs to process. Skipping.")
            continue

        output_dir = output_base_dir / category
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / f"{args.model_name}.csv"

        results_data = []
        for term1, term2 in pairs:
            if term1 in model and term2 in model:
                # Use the model's built-in similarity function
                similarity = model.similarity(term1, term2)
                results_data.append({"term_1": term1, "term_2": term2, "cosine_similarity": float(similarity)})

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        print(f"✅ Saved similarity scores for {category} to {output_path}")

    print(f"\n🎉 Glorious success! All categories for {args.model_name} have been processed!")

if __name__ == "__main__":
    main()