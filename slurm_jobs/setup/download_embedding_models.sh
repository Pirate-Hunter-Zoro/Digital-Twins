#!/bin/bash

# Activate your wonderful Conda environment!
source /opt/apps/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate dt_env

# --- THIS IS THE NEW SECRET AGENT PART! ---
# First, we find our project's home!
PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
ENV_FILE="${PROJECT_ROOT}/.env"

# Check if the secret .env file exists!
if [ -f "$ENV_FILE" ]; then
  echo "🤫 Found our secret .env file! Reading the secret codes..."
  # We read the file and export the variables, making them ready to use!
  export $(grep -v '^#' "$ENV_FILE" | xargs)
  # The hub library looks for this EXACT name, and we get it from your variable!
  export HUGGING_FACE_HUB_TOKEN=$HUGGINGFACE_TOKEN
else
  echo "⚠️  WARNING! I couldn't find a .env file with the HUGGINGFACE_TOKEN inside!"
fi


# We are serious this time! If anything fails, we stop!
set -e

# --- Get our bearings! ---
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"
DEST_DIR="/media/scratch/mferguson/models"

# Let's make sure our model home exists!
mkdir -p "$DEST_DIR"

if [ ! -f "$CANDIDATE_FILE" ]; then
    echo "❌ ERROR: I can't find the vectorizer candidates at $CANDIDATE_FILE!"
    exit 1
fi

# Read all our wonderful candidates into a list!
mapfile -t MODELS < "$CANDIDATE_FILE"

echo "📦 Found ${#MODELS[@]} models to download into our scratchpad at $DEST_DIR! Let the great download begin!"

for model_id in "${MODELS[@]}"; do
  # Make a safe folder name! No slashes allowed!
  folder_name=$(echo "$model_id" | tr '/' '-')
  model_dir="$DEST_DIR/$folder_name"

  if [ -d "$model_dir" ]; then
    echo "✅ Hooray! We already have $model_id at $model_dir"
  else
    echo "⬇️  Here comes $model_id! It's going to love its new home!"
    python -c "
from sentence_transformers import SentenceTransformer
model_id = '$model_id'
model_dir = '$model_dir'
print(f'Saving {model_id} to {model_dir}...')
model = SentenceTransformer(model_id)
model.save(model_dir)
print(f'✅ Successfully saved {model_id}!')
" || echo "⚠️  Whoopsie! Had a little trouble with $model_id. We'll skip it for now!"
  fi
done

echo "🎉 YAY! All our transformer models are downloaded and ready for the gauntlet!"