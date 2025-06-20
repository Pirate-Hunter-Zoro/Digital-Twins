import json
import pickle
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

def forge_innate_technique_library():
    # ... (the pathing logic is the same) ...
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    
    # --- This is the ONLY CHANGE ---
    # Instead of the Hugging Face name, we point to our local Cursed Tool vault!
    local_model_path = project_root / "pretrained_models" / "dmis-lab/biobert-base-cased-v1.1"

    # ... (the rest of the script from here on is almost identical) ...
    # ... except we use 'local_model_path' instead of 'model_name'
    
    data_folder = project_root / "real_data"
    input_registry_path = data_folder / "term_idf_registry.json"
    output_library_path = data_folder / "term_embedding_library.pkl"

    print("Beginning the forging of the Innate Technique Library from a LOCAL source...")
    
    if not input_registry_path.exists():
        print(f"ERROR: Cannot find the Cursed Energy Registry at {input_registry_path}. Run the first script first!")
        return

    if not local_model_path.exists():
        print(f"ERROR: The local model vault does not exist at '{local_model_path}'")
        print("Please run the 'download_model.py' script first on the login node.")
        return

    print(f"Loading the pre-summoned BioBERT model from: {local_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModel.from_pretrained(local_model_path)
    print("BioBERT model is active and ready.")

    # --- Load the list of all unique terms (our targets) ---
    print(f"Loading all unique terms from {input_registry_path.name}...")
    with open(input_registry_path, 'r', encoding='utf-8') as f:
        idf_scores = json.load(f)
    terms_to_process = list(idf_scores.keys())
    print(f"Found {len(terms_to_process)} unique terms to analyze.")

    # --- Channeling Cursed Energy to define each Innate Technique ---
    print("Beginning vectorization process. This is the longest part of the ritual...")
    embedding_library = {}
    total_terms = len(terms_to_process)

    # We go through each term one-by-one to define its technique
    for i, term in enumerate(terms_to_process):
        # Tokenize the term (break it down into pieces the model understands)
        inputs = tokenizer(term, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Pass it through the model to get the raw energy output
        with torch.no_grad(): # We don't need to train, just observe
            outputs = model(**inputs)

        # "Mean pooling" - A technique to distill the raw energy into a single, stable vector
        # This vector is the term's Innate Technique!
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Store the technique (as a numpy array) in our library
        embedding_library[term] = embedding.cpu().numpy()

        # A little progress update so we know the ritual is working
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