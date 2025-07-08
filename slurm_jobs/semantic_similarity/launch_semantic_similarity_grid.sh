#!/bin/bash

# === Embedding Models ===
VECTORIZERS=("BioBERT-mnli-snli-scinli-scitail-mednli-stsb" "all-mpnet-base-v2" "biobert-mnli-mednli" "Qwen/Qwen3-Embedding-8B")

TEMPLATE="slurm_jobs/semantic_similarity/semantic_similarity_template.ssub"

mkdir -p logs data

for VEC in "${VECTORIZERS[@]}"; do
  JOB_NAME="semantic_sim_${VEC}"
  LOG_OUT="logs/${JOB_NAME}_out.txt"
  LOG_ERR="logs/${JOB_NAME}_err.txt"

  echo "Launching $JOB_NAME..."

  sbatch \
    --job-name="$JOB_NAME" \
    --output="$LOG_OUT" \
    --error="$LOG_ERR" \
    "$TEMPLATE" "$VEC"

  sleep 1
done
