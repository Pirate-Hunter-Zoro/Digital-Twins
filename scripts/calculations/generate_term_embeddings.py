import json
import pickle
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

def forge_innate_technique_library():
    """
    Uses a pre-existing, locally stored BioBERT model to generate vector
    embeddings for each unique medical term. This version points directly
    to a known model path on the cluster.
    """
    # --- Spatial awareness to find our input/output files ---
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    data_folder = project_root / "data"
    
    input_registry_path = data_folder / "term_idf_registry.json"
    output_library_path = data_folder / "term_embedding_library.pkl"

    print("Beginning the final forging of the Innate Technique Library...")
    
    # --- Summoning the LOCAL Cursed Tool ---
    # This is the absolute, hardcoded path to YOUR existing BioBERT model.
    # No more complex pathfinding needed for the model itself!
    local_model_path = "/home/librad.laureateinstitute.org/mferguson/models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

    print(f"Loading the pre-existing BioBERT model from: {local_model_path}")

    # The all-important binding vow: local_files_only=True
    # This forbids the script from trying to access the internet.
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
    
    print("BioBERT model is active and ready.")

    # --- Load the list of all unique terms (our targets) ---
    print(f"Loading all unique terms from {input_registry_path.name}...")
    with open(input_registry_path, 'r', encoding='utf-8') as f:
        idf_scores = json.load(f)
    terms_to_process = list(idf_scores.keys())
    print(f"Found {len(terms_to_process)} unique terms to analyze.")

    # --- Channeling Cursed Energy to define each Innate Technique ---
    print("Beginning vectorization process...")
    embedding_library = {}
    total_terms = len(terms_to_process)

    for i, term in enumerate(terms_to_process):
        inputs = tokenizer(term, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        embedding_library[term] = embedding.cpu().numpy()

        if (i + 1) % 100 == 0:
            print(f"  ...processed {i + 1} / {total_terms} terms...")

    print("All Innate Techniques have been defined.")

    # --- Sealing the library into a powerful pickle file ---
    print(f"Sealing the Technique Library into: {output_library_path}")
    with open(output_library_path, 'wb') as f:
        pickle.dump(embedding_library, f)

    print("\nRitual complete! The Innate Technique Library has been successfully forged!")


if __name__ == "__main__":
    forge_innate_technique_library()