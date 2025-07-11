#!/bin/bash

# The only job of this script is to LAUNCH other jobs!
set -e

# --- The Magic Part! Makes our script self-aware! ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Now we can find the project root from wherever we are! So smart!
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../../")
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/legacy_similarity_template.ssub"


# --- Our Legacy Contestants! ---
declare -A MODELS
MODELS["glove-6B-300d"]="/media/scratch/mferguson/legacy_models/glove.6B.300d.txt"
MODELS["word2vec-google-news-300"]="/media/scratch/mferguson/legacy_models/GoogleNews-vectors-negative300.bin"

# A quick check to make sure our magic worked!
if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find my beautiful submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the Games Begin! ---
for model_name in "${!MODELS[@]}"; do
    model_path=${MODELS[$model_name]}
    
    echo "🚀 Submitting job for ${model_name}!"
    # The sbatch command sends our worker bot off to the queue!
    sbatch "$SBATCH_TEMPLATE_PATH" "$model_name" "$model_path"
done

echo "🎉 All legacy jobs have been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on them!"