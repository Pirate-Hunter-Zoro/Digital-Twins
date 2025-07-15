#!/bin/bash
LOG_DIR="slurm_jobs/world_4_embedder_gauntlet/logs"
mkdir -p "$LOG_DIR"

MODEL_NAMES=("glove_legacy" "word2vec_legacy")
MODEL_PATHS=("/media/scratch/mferguson/legacy_models/glove.6B.300d.txt" "/media/scratch/mferguson/legacy_models/GoogleNews-vectors-negative300.bin")
IS_WORD2VEC_FLAGS=(false true)

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[i]}"
    MODEL_PATH="${MODEL_PATHS[i]}"
    IS_WORD2VEC="${IS_WORD2VEC_FLAGS[i]}"
    JOB_NAME="W4_legacy_norm_sim_${MODEL_NAME}"

    echo "Submitting NORMALIZED legacy job: $JOB_NAME"
    export MODEL_PATH_ENV="$MODEL_PATH"
    export MODEL_NAME_ENV="$MODEL_NAME"
    export IS_WORD2VEC_ENV="$IS_WORD2VEC"

    sbatch --job-name="$JOB_NAME" \
           --output="${LOG_DIR}/${JOB_NAME}_out.txt" \
           --error="${LOG_DIR}/${JOB_NAME}_err.txt" \
           --export=ALL \
           slurm_jobs/world_4_embedder_gauntlet/legacy_normalized_similarity_template.ssub
done