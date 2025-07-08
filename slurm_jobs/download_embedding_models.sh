#!/bin/bash

# Activate Conda environment
source /opt/apps/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate hugging_env

set -e

echo "📦 Downloading GPT-style models into ../models"

# Load Hugging Face token from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "❌ .env file not found. Please create one with your HUGGINGFACE_TOKEN."
  exit 1
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
  echo "❌ HUGGINGFACE_TOKEN is not set in .env. Exiting."
  exit 1
fi

MODELS=(
"sentence-transformers/all-mpnet-base-v2"
"pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
"Qwen/Qwen3-Embedding-8B"
)

DEST_DIR="../models"

for model_id in "${MODELS[@]}"; do
  folder_name=$(echo "$model_id" | tr '/' '-')
  echo "⬇️  Downloading $model_id into $folder_name"

  python <<EOF || echo "⚠️  Skipped $model_id due to error"
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
import os

login(token=os.environ["HUGGINGFACE_TOKEN"])
model_id = "$model_id"
model_dir = os.path.join("$DEST_DIR", "$folder_name")

trust_remote_code = model_id in [
    "allenai/OLMo-7B"
]

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
tokenizer.save_pretrained(model_dir)

model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
model.save_pretrained(model_dir)
EOF

done

echo "🎉 GPT-style model downloads complete!"
