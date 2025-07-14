#!/bin/bash

echo "--- 🚀 Preparing to Launch World 2 Experiment Grid 🚀 ---"

mkdir -p slurm_output

# --- Define the grid of parameters ---
REPRESENTATION_METHODS=("visit_sentence")
VECTORIZER_METHODS=("allenai/scibert_scivocab_uncased")
DISTANCE_METRICS=("euclidean")
NUM_VISITS_LIST=(6)
NUM_PATIENTS_LIST=(5000)
NUM_NEIGHBORS_LIST=(5)


# --- Loop through all parameter combinations and launch jobs ---
for rep_method in "${REPRESENTATION_METHODS[@]}"; do
  for vec_method in "${VECTORIZER_METHODS[@]}"; do
    # Use the full vectorizer method name! No more basename!
    for dist_metric in "${DISTANCE_METRICS[@]}"; do
      for num_visits in "${NUM_VISITS_LIST[@]}"; do
        for num_patients in "${NUM_PATIENTS_LIST[@]}"; do
          for num_neighbors in "${NUM_NEIGHBORS_LIST[@]}"; do

            # Create a unique job name for each experiment!
            JOB_NAME="W2_${rep_method}_${vec_method//\//-}_${dist_metric}_v${num_visits}_p${num_patients}_n${num_neighbors}"

            echo "--------------------------------------------------"
            echo "Submitting Job: $JOB_NAME"
            echo "--------------------------------------------------"

            # ✨ Export variables and submit the job with explicit output paths! ✨
            export REP_METHOD_ENV="$rep_method"
            export VEC_METHOD_ENV="$vec_method" # Pass the full name!
            export DIST_METRIC_ENV="$dist_metric"
            export NUM_VISITS_ENV="$num_visits"
            export NUM_PATIENTS_ENV="$num_patients"
            export NUM_NEIGHBORS_ENV="$num_neighbors"

            sbatch --job-name="$JOB_NAME" \
                   --output="slurm_output/${JOB_NAME}_out.txt" \
                   --error="slurm_output/${JOB_NAME}_err.txt" \
                   --export=ALL \
                   slurm_jobs/world_2_neighbor_analysis/run_world2_analysis.ssub

          done
        done
      done
    done
  done
done

echo "\n--- ✅ All jobs launched! Monitor with 'squeue -u your_username' ---"