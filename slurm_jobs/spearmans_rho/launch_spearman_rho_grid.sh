#!/bin/bash
# launch_spearman_grid.sh

REPRESENTATIONS=("visit_sentence")
NUM_PATIENTS_LIST=(5000)
NUM_VISITS_LIST=(6)
VECTORIZERS=("BioBERT-mnli-snli-scinli-scitail-mednli-stsb" "all-mpnet-base-v2" "biobert-mnli-mednli")
DISTANCE_METRICS=("euclidean")
NUM_NEIGHBORS_LIST=(5)
MAX_PATIENTS_LIST=(500)
MODELS=("medgemma")
MODEL_PATHS=("unsloth/medgemma-27b-text-it-bnb-4bit")

mkdir -p logs

for REP in "${REPRESENTATIONS[@]}"; do
  for PTS in "${NUM_PATIENTS_LIST[@]}"; do
    for VISITS in "${NUM_VISITS_LIST[@]}"; do
      for VEC in "${VECTORIZERS[@]}"; do
        for DIST in "${DISTANCE_METRICS[@]}"; do
          for NBR in "${NUM_NEIGHBORS_LIST[@]}"; do
            for MAXP in "${MAX_PATIENTS_LIST[@]}"; do
              for i in "${!MODELS[@]}"; do
                MODEL_NAME="${MODELS[$i]}"
                MODEL_PATH="${MODEL_PATHS[$i]}"

                sbatch \
                  --export=ALL,REP=$REP,PTS=$PTS,VISITS=$VISITS,VEC=$VEC,DIST=$DIST,NBR=$NBR,MAXP=$MAXP,MODEL_NAME=$MODEL_NAME,MODEL_PATH=$MODEL_PATH \
                  slurm_jobs/spearmans_rho/spearman_grid_template.ssub

              done
            done
          done
        done
      done
    done
  done
done
