
import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

def forge_innate_technique_library():
    """
    Uses the all-mpnet-base-v2 model to generate sentence-level embeddings
    for each unique medical term in the IDF registry.
    """
    # --- Locate project paths ---
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    data_folder = project_root / "data"

    input_registry_path = data_folder / "term_idf_registry.json"
    output_library_path = data_folder / "term_embedding_library_mpnet.pkl"

    print("🧬 Beginning the final forging of the Innate Technique Library (MPNet Edition)...")

    # --- Load local SentenceTransformer model from cache ---
    local_model_path = "/home/librad.laureateinstitute.org/mferguson/models/all-mpnet-base-v2"
    print(f"🧠 Loading model from: {local_model_path}")
    model = SentenceTransformer(local_model_path)

    # --- Load the list of unique terms to embed ---
    print(f"📚 Reading terms from: {input_registry_path.name}")
    with open(input_registry_path, 'r', encoding='utf-8') as f:
        idf_scores = json.load(f)
    terms_to_process = list(idf_scores.keys())
    print(f"🔢 Total terms to embed: {len(terms_to_process)}")

    # --- Generate embeddings ---
    print("💥 Vectorizing terms...")
    embeddings = model.encode(terms_to_process, batch_size=64, show_progress_bar=True)

    # --- Save the final library ---
    print(f"💾 Saving new embedding library to: {output_library_path.name}")
    embedding_library = {term: vec for term, vec in zip(terms_to_process, embeddings)}
    with open(output_library_path, 'wb') as f:
        pickle.dump(embedding_library, f)

    print("✅ Library generation complete! MPNet powers activated!")

if __name__ == "__main__":
    forge_innate_technique_library()
