#!/bin/bash

# This script should be placed in Digital-Twins/slurm_jobs
# It downloads various sentence embedding models to ../models/

# Navigate to the project parent directory (one level above Digital-Twins)
cd "$(dirname "$0")/../.."

# Ensure models directory exists
mkdir -p models

# Array of HuggingFace model IDs to download
MODEL_LIST=(
  "sentence-transformers/all-mpnet-base-v2"
  "sentence-transformers/all-MiniLM-L6-v2"
  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
  "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
)

echo "Downloading models to: $(pwd)/models"

for MODEL_ID in "${MODEL_LIST[@]}"; do
  DIR_NAME=$(echo "$MODEL_ID" | sed 's|/|-|g')
  TARGET_DIR="models/$DIR_NAME"

  if [ -d "$TARGET_DIR" ]; then
    echo "✅ Model already exists: $DIR_NAME"
  else
    echo "⬇️  Downloading $MODEL_ID into $DIR_NAME using SentenceTransformer"
    python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('${MODEL_ID}')
model.save('${TARGET_DIR}')
" || echo "❌ Failed to download $MODEL_ID"
  fi
done

echo "🎉 Done downloading all sentence embedding models!"
