#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/semantic_similarity_template.ssub"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"


# --- Get our list of wonderful candidates! ---
PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"

if [ ! -f "$CANDIDATE_FILE" ]; then
    echo "ERROR! I can't find the vectorizer candidates file at $CANDIDATE_FILE"
    exit 1
fi

mapfile -t MODELS < "$CANDIDATE_FILE"

if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find my beautiful submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the GAUNTLET Begin! ---
for model_name in "${MODELS[@]}"; do
    # ✨ Create a safe and descriptive job name ✨
    JOB_NAME="W4_sim_${model_name//\//-}"

    echo "🚀 Submitting job for ${model_name}!"
    
    # ✨ NEW: Requesting more power directly in the sbatch command! ✨
    sbatch --job-name="$JOB_NAME" \
           --output="${LOG_DIR}/${JOB_NAME}_out.txt" \
           --error="${LOG_DIR}/${JOB_NAME}_err.txt" \
           --gres=gpu:2 \
           --cpus-per-task=36 \
           "$SBATCH_TEMPLATE_PATH" "$model_name"
done

echo "🎉 All jobs have been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on them!"