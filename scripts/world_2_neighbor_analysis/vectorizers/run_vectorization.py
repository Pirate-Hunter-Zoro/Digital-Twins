import os
import sys
import argparse
import pickle
from pathlib import Path

# --- ✨ THE MAGNIFICENT RE-FIX! ✨ ---
# This little contraption tells the script where the project's home is!
current_script_dir = Path(__file__).resolve().parent
# We go up THREE whole levels to get to the project root! You were right!
project_root = current_script_dir.parents[2] 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ------------------------------------

from scripts.common.data_loading.load_patient_data import load_patient_data
from scripts.common.config import setup_config, get_global_config
# --- ✨ Correctly importing our magnificent, refactored machines! ✨ ---
from scripts.common.models.gru_embedder import GRUEmbedder
from scripts.common.models.transformer_embedder import TransformerEmbedder

def main():
    print("🏭 Welcome to the Magnificent Vectorization Factory! 🏭")
    parser = argparse.ArgumentParser(description="A scalable factory for creating different patient vector embeddings.")
    parser.add_argument("--embedder_type", required=True, choices=["gru", "transformer"], help="The type of embedder to use.")
    parser.add_argument("--num_visits", type=int, default=6)
    args = parser.parse_args()

    setup_config("visit_sentence", "placeholder", "cosine", args.num_visits, 5000, 5)
    config = get_global_config()

    output_dir = project_root / "data" / "visit_sentence" / f"visits_{config.num_visits}" / "patients_5000"
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_output_path = output_dir / f"all_vectors_{args.embedder_type}.pkl"

    if vector_output_path.exists():
        print(f"✅ Vectors for embedder '{args.embedder_type}' already exist! Nothing to do!")
        return

    print("📂 Loading patient data...")
    patient_data = load_patient_data()
    
    if args.embedder_type == "gru":
        print("🤖 Powering up the GRU Embedder machine...")
        embedder = GRUEmbedder()
        vectors_dict = embedder.vectorize(patient_data, config)

    elif args.embedder_type == "transformer":
        print("🤖 Powering up the mighty Transformer Embedder machine...")
        embedder = TransformerEmbedder()
        vectors_dict = embedder.vectorize(patient_data, config)
    
    else:
        raise ValueError("Unknown embedder type! The factory is confused!")

    print(f"\n💾 Saving {len(vectors_dict)} vectors to: {vector_output_path}")
    with open(vector_output_path, "wb") as f:
        pickle.dump(vectors_dict, f)
    
    print("🎉 Vectorization complete! Another successful day at the factory!")

if __name__ == "__main__":
    main()