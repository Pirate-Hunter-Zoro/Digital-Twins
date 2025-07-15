#!/bin/bash

echo "--- 🚀 Preparing to Launch LEGACY Model Gauntlet 🚀 ---"

LOG_DIR="slurm_jobs/world_4_embedder_gauntlet/logs"
mkdir -p "$LOG_DIR"

# --- Define our magnificent legacy models and their properties! ---
MODEL_NAMES=("glove_legacy" "word2vec_legacy")
MODEL_PATHS=("/media/scratch/mferguson/legacy_models/glove.6B.300d.txt" "/media/scratch/mferguson/legacy_models/GoogleNews-vectors-negative300.bin")
IS_WORD2VEC_FLAGS=(false true)

# --- Loop through our legacy models ---
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[i]}"
    MODEL_PATH="${MODEL_PATHS[i]}"
    IS_WORD2VEC="${IS_WORD2VEC_FLAGS[i]}"

    JOB_NAME="W4_legacy_sim_${MODEL_NAME}"

    echo "--------------------------------------------------"
    echo "Submitting Job: $JOB_NAME"
    echo "  Model Path: $MODEL_PATH"
    echo "  Is Word2Vec?: $IS_WORD2VEC"
    echo "--------------------------------------------------"

    # Export variables for the job
    export MODEL_PATH_ENV="$MODEL_PATH"
    export MODEL_NAME_ENV="$MODEL_NAME"
    export IS_WORD2VEC_ENV="$IS_WORD2VEC"

    sbatch --job-name="$JOB_NAME" \
           --output="${LOG_DIR}/${JOB_NAME}_out.txt" \
           --error="${LOG_DIR}/${JOB_NAME}_err.txt" \
           --export=ALL \
           slurm_jobs/world_4_embedder_gauntlet/legacy_similarity_template.ssub
done

echo "\n--- ✅ All legacy jobs launched! ---"