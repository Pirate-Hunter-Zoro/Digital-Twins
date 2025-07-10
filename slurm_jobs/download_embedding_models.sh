#!/bin/bash

# Activate your wonderful Conda environment!
source /opt/apps/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate hugging_env

set -e

# Find our project root, because we're smart like that!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"
# Our NEW, ULTIMATE, SCRATCH-TASTIC model destination!
DEST_DIR="/media/scratch/mferguson/models"

# Make sure our new home exists!
mkdir -p "$DEST_DIR"

if [ ! -f "$CANDIDATE_FILE" ]; then
    echo "❌ ERROR: I can't find the vectorizer candidates at $CANDIDATE_FILE!"
    exit 1
fi

# Read all our model friends into a list!
mapfile -t MODELS < "$CANDIDATE_FILE"

echo "📦 Found ${#MODELS[@]} models to download into our ultimate scratchpad at $DEST_DIR! Let the downloading BEGIN!"

for model_id in "${MODELS[@]}"; do
  # Make a safe folder name! No slashes allowed!
  folder_name=$(echo "$model_id" | tr '/' '-')
  model_dir="$DEST_DIR/$folder_name"

  if [ -d "$model_dir" ]; then
    echo "✅ Hooray! We already have $model_id at $model_dir"
  else
    echo "⬇️  Here comes $model_id! ZOOOOOM!"
    # Use a little Python magic to download!
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

echo "🎉 YAY! All the models are snug in their new scratchy homes!"