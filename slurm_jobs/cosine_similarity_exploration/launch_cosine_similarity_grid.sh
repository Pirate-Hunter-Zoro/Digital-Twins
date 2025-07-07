#!/bin/bash

# === Grid Parameters ===
REPRESENTATIONS=("visit_sentence")
VECTORIZERS=("biobert-mnli-mednli" "gte-large" "e5-large-v2" "all-mpnet-base-v2" "mxbai-embed-large-v1")
DISTANCE_METRICS=("euclidean")
NUM_PATIENTS_LIST=(5000)
NUM_VISITS_LIST=(6)
MODEL_NAMES=("medgemma")

TEMPLATE="slurm_jobs/cosine_similarity_exploration/cosine_similarity_exploration.ssub"

mkdir -p logs data

for REP in "${REPRESENTATIONS[@]}"; do
  for VEC in "${VECTORIZERS[@]}"; do
    for DIST in "${DISTANCE_METRICS[@]}"; do
      for PATIENTS in "${NUM_PATIENTS_LIST[@]}"; do
        for VISITS in "${NUM_VISITS_LIST[@]}"; do
          for MODEL_NAME in "${MODEL_NAMES[@]}"; do

            JOB_NAME="cosine_${REP}_${VEC}_${DIST}_${PATIENTS}_${VISITS}"
            LOG_OUT="logs/${JOB_NAME}_out.txt"
            LOG_ERR="logs/${JOB_NAME}_err.txt"

            echo "Launching $JOB_NAME..."

            sbatch \
              --job-name="$JOB_NAME" \
              --output="$LOG_OUT" \
              --error="$LOG_ERR" \
              "$TEMPLATE" "$REP" "$VEC" "$DIST" "$PATIENTS" "$VISITS" "$MODEL_NAME"

            sleep 1
          done
        done
      done
    done
  done
done
