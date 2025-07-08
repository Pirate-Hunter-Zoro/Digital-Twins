#!/bin/bash
# launch_predict_grid.sh

REPRESENTATIONS=("visit_sentence")
NUM_PATIENTS_LIST=(5000)
NUM_VISITS_LIST=(6)
VECTORIZERS=("BioBERT-mnli-snli-scinli-scitail-mednli-stsb" "all-mpnet-base-v2" "biobert-mnli-mednli" "Qwen/Qwen3-Embedding-8B")
DISTANCE_METRICS=("euclidean")
NUM_NEIGHBORS_LIST=(5)
MODELS=("medgemma")
MODEL_PATHS=("unsloth/medgemma-27b-text-it-bnb-4bit")

mkdir -p logs

for REP in "${REPRESENTATIONS[@]}"; do
  for PTS in "${NUM_PATIENTS_LIST[@]}"; do
    for VISITS in "${NUM_VISITS_LIST[@]}"; do
      for VEC in "${VECTORIZERS[@]}"; do
        for DIST in "${DISTANCE_METRICS[@]}"; do
          for NBR in "${NUM_NEIGHBORS_LIST[@]}"; do
            for i in "${!MODELS[@]}"; do
              MODEL_NAME="${MODELS[$i]}"
              MODEL_PATH="${MODEL_PATHS[$i]}"

              sbatch \
                --export=ALL,REP=$REP,PTS=$PTS,VISITS=$VISITS,VEC=$VEC,DIST=$DIST,NBR=$NBR,MODEL_NAME=$MODEL_NAME,MODEL_PATH=$MODEL_PATH \
                slurm_jobs/next_visit_prediction/predict_grid_template.ssub
            done
          done
        done
      done
    done
  done
done
