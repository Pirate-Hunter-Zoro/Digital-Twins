#!/bin/bash

REPRESENTATIONS=("visit_sentence")
NUM_PATIENTS_LIST=(5000)
NUM_VISITS_LIST=(6)
VECTORIZERS=("BioBERT-mnli-snli-scinli-scitail-mednli-stsb" "all-MiniLM-L6-v2" "all-mpnet-base-v2" "biobert-mnli-mednli" "cambridgeltl-SapBERT-from-PubMedBERT-fulltext" "paraphrase-multilingual-MiniLM-L12-v2" "pritamdeka-BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
DISTANCE_METRICS=("euclidean")
NUM_NEIGHBORS_LIST=(5)
MODELS=("medgemma")

mkdir -p logs data

for VECTOR in "${VECTORIZERS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for REP in "${REPRESENTATIONS[@]}"; do
      for NUM_PATIENTS in "${NUM_PATIENTS_LIST[@]}"; do
        for NUM_VISITS in "${NUM_VISITS_LIST[@]}"; do
          for DIST in "${DISTANCE_METRICS[@]}"; do
            for NN in "${NUM_NEIGHBORS_LIST[@]}"; do
              sbatch slurm_jobs/idf_grid/generate_idf_registry_template.ssub \
                "$VECTOR" "$MODEL" "$REP" "$NUM_PATIENTS" "$NUM_VISITS" "$DIST" "$NN"
            done
          done
        done
      done
    done
  done
done
