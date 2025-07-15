#!/bin/bash

echo "--- 🚀 Preparing to Launch World 2 Experiment Grid 🚀 ---"

# --- ✨ NEW: Define and create our new, super-organized log directory! ✨ ---
LOG_DIR="slurm_jobs/world_2_neighbor_analysis/logs"
mkdir -p "$LOG_DIR"


# --- Define the grid of parameters ---
REPRESENTATION_METHODS=("visit_sentence")
VECTORIZER_METHODS=("allenai/scibert_scivocab_uncased")
DISTANCE_METRICS=("euclidean")
NUM_VISITS_LIST=(6)
NUM_PATIENTS_LIST=(5000)
NUM_NEIGHBORS_LIST=(5)
BATCH_SIZES=(1000)


# --- Loop through all parameter combinations and launch jobs ---
for rep_method in "${REPRESENTATION_METHODS[@]}"; do
  for vec_method in "${VECTORIZER_METHODS[@]}"; do
    vec_method_name_for_path="${vec_method//\//-}"
    for dist_metric in "${DISTANCE_METRICS[@]}"; do
      for num_visits in "${NUM_VISITS_LIST[@]}"; do
        for num_patients in "${NUM_PATIENTS_LIST[@]}"; do
          for num_neighbors in "${NUM_NEIGHBORS_LIST[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do

              JOB_NAME="W2_${rep_method}_${vec_method_name_for_path}_${dist_metric}_v${num_visits}_p${num_patients}_n${num_neighbors}_b${batch_size}"

              echo "--------------------------------------------------"
              echo "Submitting Job: $JOB_NAME"
              echo "--------------------------------------------------"

              # Export variables for the job
              export REP_METHOD_ENV="$rep_method"
              export VEC_METHOD_ENV="$vec_method"
              export DIST_METRIC_ENV="$dist_metric"
              export NUM_VISITS_ENV="$num_visits"
              export NUM_PATIENTS_ENV="$num_patients"
              export NUM_NEIGHBORS_ENV="$num_neighbors"
              export BATCH_SIZE_ENV="$batch_size"
              
              # ✨ The final sbatch command with perfect log paths! ✨
              sbatch --job-name="$JOB_NAME" \
                     --output="${LOG_DIR}/${JOB_NAME}_out.txt" \
                     --error="${LOG_DIR}/${JOB_NAME}_err.txt" \
                     --export=ALL \
                     slurm_jobs/world_2_neighbor_analysis/run_world2_analysis.ssub

            done
          done
        done
      done
    done
  done
done

echo "\n--- ✅ All jobs launched! Monitor with 'squeue -u your_username' ---"