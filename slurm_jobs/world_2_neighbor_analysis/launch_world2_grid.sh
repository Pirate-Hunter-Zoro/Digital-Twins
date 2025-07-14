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
  for vec_method_path in "${VECTORIZER_METHODS[@]}"; do
    vec_method_name=$(basename "$vec_method_path")
    for dist_metric in "${DISTANCE_METRICS[@]}"; do
      for num_visits in "${NUM_VISITS_LIST[@]}"; do
        for num_patients in "${NUM_PATIENTS_LIST[@]}"; do
          for num_neighbors in "${NUM_NEIGHBORS_LIST[@]}"; do

            # ✨ Create a unique job name for each experiment! ✨
            JOB_NAME="W2_${rep_method}_${vec_method_name}_${dist_metric}_v${num_visits}_p${num_patients}_n${num_neighbors}"

            echo "--------------------------------------------------"
            echo "Submitting Job: $JOB_NAME"
            echo "--------------------------------------------------"

            # Use sbatch to submit the job, passing the JOB_NAME and all other parameters
            sbatch slurm_jobs/world_2_neighbor_analysis/run_world2_analysis.ssub \
              "$JOB_NAME" \
              "$rep_method" \
              "$vec_method_name" \
              "$dist_metric" \
              "$num_visits" \
              "$num_patients" \
              "$num_neighbors"
              
          done
        done
      done
    done
  done
done

echo "\n--- ✅ All jobs launched! Monitor with 'squeue -u your_username' ---"