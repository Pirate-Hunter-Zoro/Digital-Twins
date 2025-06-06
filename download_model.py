from huggingface_hub import snapshot_download
import os

# Ensure TOKENIZERS_PARALLELISM is set to avoid warnings, often helpful on clusters
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Download Meta-Llama-3.1-8B-Instruct ---
# Make sure you replace 'YOUR_HUGGING_FACE_TOKEN_HERE' with your actual token
print("Attempting to download Meta-Llama-3.1-8B-Instruct...")
try:
    snapshot_download(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        local_dir="/home/librad.laureateinstitute.org/mferguson/models/Meta-Llama",
        local_dir_use_symlinks=False,
        token="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" # Replace with your actual token
    )
    print("Meta-Llama-3.1-8B-Instruct downloaded successfully.")
except Exception as e:
    print(f"Error downloading Meta-Llama-3.1-8B-Instruct: {e}")

# NEW: Download MedGemma-27B-text-it-GGUF (quantized)
medgemma_gguf_repo_id = "unsloth/medgemma-27b-text-it-GGUF"
medgemma_gguf_filename = "medgemma-27b-text-it-q4_k_m.gguf" # Specify the exact file to download
medgemma_gguf_local_dir = "/home/librad.laureateinstitute.org/mferguson/models/MedGemma-27B-text-it-GGUF" # New directory for this model
print(f"\nAttempting to download MedGemma-27B-text-it-GGUF ({medgemma_gguf_filename})...")
try:
    # Use allow_patterns to download only the specific GGUF file
    snapshot_download(
        repo_id=medgemma_gguf_repo_id,
        local_dir=medgemma_gguf_local_dir,
        allow_patterns=[medgemma_gguf_filename],
        local_files_only=False,
        token="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"
    )
    print(f"MedGemma-27B-text-it-GGUF ({medgemma_gguf_filename}) downloaded successfully to {medgemma_gguf_local_dir}.")
except Exception as e:
    print(f"Error downloading MedGemma-27B-text-it-GGUF: {e}", exc_info=True)