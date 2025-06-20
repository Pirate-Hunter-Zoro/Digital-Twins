from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def perform_pre_summoning():
    """
    A dedicated ritual to download all model and tokenizer files from Hugging Face
    and save them locally. This bypasses network restrictions on compute nodes.
    This version is location-aware to be run from the 'scripts' folder.
    """
    # --- Spatial awareness technique ---
    # Find the script's own location
    script_path = Path(__file__).resolve()
    # Navigate up to the project root (from 'scripts/')
    project_root = script_path.parent.parent
    
    # Define the model and the output path relative to the project root
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    output_path = project_root / "pretrained_models" / model_name

    print(f"Preparing to summon '{model_name}' into local vault: {output_path}")

    # This will download and cache all necessary files
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Now, we save them from the cache into our specified vault
    # The 'exist_ok=True' is important so we don't get an error if the folder is already there
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print("\nPre-summoning complete!")
    print(f"The essence of BioBERT has been successfully sealed in '{output_path}'")
    print("You can now use this local path in the main script.")

if __name__ == "__main__":
    perform_pre_summoning()