#!/bin/bash
set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/semantic_similarity_template.ssub"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"
mapfile -t MODELS < "$CANDIDATE_FILE"

for model_name in "${MODELS[@]}"; do
    JOB_NAME="W4_sim_${model_name//\//-}"
    echo "🚀 Submitting job for ${model_name}!"
    
    # It now uses the variables passed down from the master launcher!
    sbatch --job-name="$JOB_NAME" \
           --output="${LOG_DIR}/${JOB_NAME}_out.txt" \
           --error="${LOG_DIR}/${JOB_NAME}_err.txt" \
           --gres=gpu:"$GPU_COUNT_ENV" \
           --cpus-per-task="$CPU_COUNT_ENV" \
           "$SBATCH_TEMPLATE_PATH" "$model_name"
done