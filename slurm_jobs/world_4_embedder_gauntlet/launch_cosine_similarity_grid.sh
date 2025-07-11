#!/bin/bash
# === New GPT Embedding Model Download Section ===

# Root path for model downloads
MODEL_ROOT=../models
mkdir -p $MODEL_ROOT

echo "📦 Downloading GPT-style models into $MODEL_ROOT"

GPT_MODELS=(
  "EleutherAI/gpt-j-6B"
  "mistralai/Mistral-7B-Instruct-v0.2"
  "NousResearch/Nous-Hermes-2-Mistral-7B"
  "openchat/openchat-3.5-1210"
  "openai-community/gpt2"  # optionally test classic GPT-2
  "allenai/OLMo-7B"        # from the paper, a newer high-quality model
)

for MODEL in "${GPT_MODELS[@]}"; do
  MODEL_DIR=${MODEL//\//-}
  TARGET_DIR="$MODEL_ROOT/$MODEL_DIR"
  if [ -d "$TARGET_DIR" ]; then
    echo "✅ Model already exists: $MODEL_DIR"
  else
    echo "⬇️  Downloading $MODEL into $MODEL_DIR"
    python3 -c "
from transformers import AutoModel, AutoTokenizer
AutoTokenizer.from_pretrained('$MODEL', cache_dir='$TARGET_DIR')
AutoModel.from_pretrained('$MODEL', cache_dir='$TARGET_DIR')
"
  fi
done

echo "🎉 GPT-style model downloads complete!"
