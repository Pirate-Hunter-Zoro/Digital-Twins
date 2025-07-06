#!/bin/bash

VECTORIZERS=("biobert-mnli-mednli")

mkdir -p logs

for VEC in "${VECTORIZERS[@]}"; do
  sbatch --export=ALL,VEC=$VEC slurm_jobs/cosine_similarity_exploration/cosine_similarity_exploration.ssub
done
