#!/bin/bash
# launch_nn_grid.sh

REPRESENTATIONS=("visit_sentence")
NUM_PATIENTS_LIST=(5000)
NUM_VISITS_LIST=(6)
VECTORIZERS=("biobert-mnli-mednli")
DISTANCE_METRICS=("euclidean")
NUM_NEIGHBORS_LIST=(10)

NUM_WORKERS=60

mkdir -p logs

for REP in "${REPRESENTATIONS[@]}"; do
  for PTS in "${NUM_PATIENTS_LIST[@]}"; do
    for VISITS in "${NUM_VISITS_LIST[@]}"; do
      for VEC in "${VECTORIZERS[@]}"; do
        for DIST in "${DISTANCE_METRICS[@]}"; do
          for NBR in "${NUM_NEIGHBORS_LIST[@]}"; do

            sbatch \
              --export=ALL,REP=$REP,PTS=$PTS,VISITS=$VISITS,VEC=$VEC,DIST=$DIST,NBR=$NBR,NUM_WORKERS=$NUM_WORKERS \
              slurm_jobs/nearest_neighbors/nn_grid_template.ssub

          done
        done
      done
    done
  done
done
