#!/bin/bash
# slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh

echo "--- 🚀 Preparing to Launch the FULL World 2 Analysis Pipeline Grid! 🚀 ---"

LOG_DIR="slurm_jobs/world_2_neighbor_analysis/logs"
mkdir -p "$LOG_DIR"

# --- ✨ Define the Grid of Parameters! ✨ ---
VECTORIZER_METHODS=("allenai/scibert_scivocab_uncased")
NUM_VISITS_LIST=(6)
LEARNING_RATES=(0.0001)

# --- ✨ NEW! Scalable Hardware Specifications! ✨ ---
GPU_COUNTS=(1)
CPUS_PER_TASK=(18)


# --- Loop through all parameter combinations and launch jobs ---
for VEC_METHOD in "${VECTORIZER_METHODS[@]}"; do
  for NUM_VISITS in "${NUM_VISITS_LIST[@]}"; do
    for LR in "${LEARNING_RATES[@]}"; do
      for GPU_COUNT in "${GPU_COUNTS[@]}"; do
        for CPU_COUNT in "${CPUS_PER_TASK[@]}"; do
    
          VEC_METHOD_NAME_FOR_PATH="${VEC_METHOD//\//-}"
          JOB_NAME="W2_VEC-${VEC_METHOD_NAME_FOR_PATH}_V-${NUM_VISITS}_LR-${LR}_G-${GPU_COUNT}_C-${CPU_COUNT}"

          echo "--------------------------------------------------"
          echo "Submitting Job: $JOB_NAME"
          echo "  GPUs: $GPU_COUNT, CPUs: $CPU_COUNT"
          echo "--------------------------------------------------"

          export VEC_METHOD_ENV="$VEC_METHOD"
          export NUM_VISITS_ENV="$NUM_VISITS"
          export LR_ENV="$LR"
          
          # This sbatch command now includes our magnificent hardware requests!
          sbatch \
            --job-name="$JOB_NAME" \
            --output="${LOG_DIR}/${JOB_NAME}.out.txt" \
            --error="${LOG_DIR}/${JOB_NAME}.err.txt" \
            --gres=gpu:"$GPU_COUNT" \
            --cpus-per-task="$CPU_COUNT" \
            --export=ALL \
            slurm_jobs/world_2_neighbor_analysis/world2_pipeline_template.ssub
        
        done
      done
    done
  done
done

echo -e "\n--- ✅ All pipeline jobs launched! Monitor with 'squeue -u mferguson' ---"