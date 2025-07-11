#!/bin/bash

# The only job of this script is to LAUNCH other jobs!
set -e

# --- THIS IS THE MAGIC PART! ---
# This line finds the directory where THIS script lives!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Now we build a perfect, absolute path to our template!
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/semantic_similarity_template.ssub"


# --- Get our list of wonderful candidates! ---
PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"

if [ ! -f "$CANDIDATE_FILE" ]; then
    echo "ERROR! I can't find the vectorizer candidates file at $CANDIDATE_FILE"
    exit 1
fi

# Read the models into a list!
mapfile -t MODELS < "$CANDIDATE_FILE"


# A quick check to make sure our magic worked!
if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find my beautiful submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the GAUNTLET Begin! ---
for model_name in "${MODELS[@]}"; do
    echo "🚀 Submitting job for ${model_name} from my cozy home at ${SCRIPT_DIR}!"
    # The sbatch command sends our worker bot off to the queue!
    sbatch "$SBATCH_TEMPLATE_PATH" "$model_name"
done

echo "🎉 All jobs have been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on them!"