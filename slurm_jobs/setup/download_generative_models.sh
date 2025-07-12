#!/bin/bash
# download_generative_models.sh

echo "🧠 Time to download a new brain! Let's get MedGemma! 🧠"

# Activate our environment so we have all the tools!
source ~/.bashrc
conda activate hugging_env

# --- THIS IS THE NEW SECRET AGENT PART! ---
# Load our secret Hugging Face key!
PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
ENV_FILE="${PROJECT_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
  echo "🤫 Found our secret .env file! Reading the secret codes..."
  export $(grep -v '^#' "$ENV_FILE" | xargs)
  export HUGGING_FACE_HUB_TOKEN=$HUGGINGFACE_TOKEN
else
  echo "⚠️  WARNING! I couldn't find a .env file with the HUGGINGFACE_TOKEN!"
fi

set -e

# The model we want and where it's going to live!
MODEL_ID="unsloth/medgemma-27b-text-it-bnb-4bit"
DEST_DIR="/media/scratch/mferguson/models/unsloth-medgemma-27b-text-it-bnb-4bit"

mkdir -p "$DEST_DIR"

if [ -d "$DEST_DIR" ] && [ "$(ls -A $DEST_DIR)" ]; then
    echo "✅ Hooray! We already have $MODEL_ID at $DEST_DIR"
else
    echo "⬇️  Here comes $MODEL_ID! It's going to love its new home at $DEST_DIR!"
    # Use python to download and save the model correctly
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = '$MODEL_ID'
dest_dir = '$DEST_DIR'
print(f'Downloading and saving {model_id} to {dest_dir}...')
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(dest_dir)
tokenizer.save_pretrained(dest_dir)
print(f'✅ Successfully saved {model_id}!')
" || echo "⚠️  Whoopsie! Had a little trouble with $MODEL_ID."
fi

echo "🎉 YAY! Our new generative model is downloaded and ready for SCIENCE!"