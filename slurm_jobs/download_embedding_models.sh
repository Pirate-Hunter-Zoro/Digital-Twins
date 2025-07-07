#!/bin/bash

# This script should be placed in Digital-Twins/slurm_jobs
# It downloads various sentence embedding models to ../models/

# Navigate to the project parent directory (one level above Digital-Twins)
cd "$(dirname "$0")/../.."

# Ensure models directory exists
mkdir -p models
cd models

# Array of HuggingFace model IDs to download
MODEL_LIST=(
  "sentence-transformers/all-mpnet-base-v2"
  "sentence-transformers/all-MiniLM-L6-v2"
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
  "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

echo "Downloading models to: $(pwd)"
for MODEL_ID in "${MODEL_LIST[@]}"; do
  # Convert model ID to folder-friendly name
  DIR_NAME=$(echo "$MODEL_ID" | sed 's|/|-|g')
  if [ -d "$DIR_NAME" ]; then
    echo "✅ Model already exists: $DIR_NAME"
  else
    echo "⬇️  Downloading $MODEL_ID into $DIR_NAME"
    transformers-cli download "$MODEL_ID" --cache-dir "./$DIR_NAME"
  fi
done

echo "🎉 Done downloading all sentence embedding models!"
