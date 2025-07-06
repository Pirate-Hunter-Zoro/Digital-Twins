#!/bin/bash
# launch_debug_prompt_grid.sh

REPRESENTATIONS=("visit_sentence")
VECTORIZERS=("biobert-mnli-mednli")
DISTANCE_METRICS=("euclidean")
NUM_VISITS_LIST=(6)
NUM_PATIENTS_LIST=(5000)
NUM_NEIGHBORS_LIST=(5)
MODEL_NAMES=("medgemma")
MODEL_PATHS=("unsloth/medgemma-27b-text-it-bnb-4bit")

mkdir -p logs

for i in "${!MODEL_NAMES[@]}"; do
  MODEL_NAME="${MODEL_NAMES[$i]}"
  MODEL_PATH="${MODEL_PATHS[$i]}"

  for REP in "${REPRESENTATIONS[@]}"; do
    for VEC in "${VECTORIZERS[@]}"; do
      for DIST in "${DISTANCE_METRICS[@]}"; do
        for VISITS in "${NUM_VISITS_LIST[@]}"; do
          for PTS in "${NUM_PATIENTS_LIST[@]}"; do
            for NEIGHBORS in "${NUM_NEIGHBORS_LIST[@]}"; do

              sbatch \
                --export=ALL,REP=$REP,VEC=$VEC,DIST=$DIST,VISITS=$VISITS,PTS=$PTS,NEIGHBORS=$NEIGHBORS,MODEL_NAME=$MODEL_NAME,MODEL_PATH=$MODEL_PATH \
                slurm_jobs/debug_visit_prompts/debug_prompt_test_template.ssub

            done
          done
        done
      done
    done
  done
done
