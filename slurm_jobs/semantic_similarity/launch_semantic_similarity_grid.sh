#!/bin/bash

# The only job of this script is to LAUNCH other jobs!
set -e

# --- Our Legacy Contestants! ---
declare -A MODELS
MODELS["glove-6B-300d"]="/media/scratch/mferguson/legacy_models/glove.6B.300d.txt"
MODELS["word2vec-google-news-300"]="/media/scratch/mferguson/legacy_models/GoogleNews-vectors-negative300.bin"

# --- Point to our NEW, perfectly named template! ---
SBATCH_TEMPLATE_PATH="$(pwd)/legacy_similarity_template.ssub"
if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find the submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the Games Begin! ---
for model_name in "${!MODELS[@]}"; do
    model_path=${MODELS[$model_name]}
    
    echo "🚀 Submitting job for ${model_name}..."
    sbatch "$SBATCH_TEMPLATE_PATH" "$model_name" "$model_path"
done

echo "🎉 All legacy jobs have been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on them!"