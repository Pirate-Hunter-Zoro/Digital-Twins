#!/bin/bash
# slurm_jobs/world_2_neighbor_analysis/launch_world2_pipeline_grid.sh

echo "--- 🚀 Preparing to Launch the SCALABLE & SAFE World 2 Analysis Pipeline Grid! 🚀 ---"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/logs"
CANCEL_SCRIPT_PATH="${SCRIPT_DIR}/cancel_world2_pipeline.sh" # Our new emergency stop button!
mkdir -p "$LOG_DIR"

# --- Create the cancel script from scratch for this run! ---
echo "#!/bin/bash" > "$CANCEL_SCRIPT_PATH"
echo "echo '--- 🛑 Sending stop signal to all World 2 pipeline jobs! 🛑 ---'" >> "$CANCEL_SCRIPT_PATH"
chmod +x "$CANCEL_SCRIPT_PATH"
echo "✅ Created new cancellation script at ${CANCEL_SCRIPT_PATH}"


# --- ✨ Define Our Magnificent, SCALABLE Grid of Parameters! ✨ ---
EMBEDDER_TYPES=("gru" "transformer")
NUM_VISITS_LIST=(6)
SAMPLE_SIZE_LIST=(50) # How many patients to use for the final plots!

# --- ✨ Hardware Specifications! ✨ ---
GPU_COUNT=1
CPU_COUNT=18

# --- Loop through ALL our magnificent parameters! ---
for NUM_VISITS in "${NUM_VISITS_LIST[@]}"; do
  for SAMPLE_SIZE in "${SAMPLE_SIZE_LIST[@]}"; do
    for EMBEDDER in "${EMBEDDER_TYPES[@]}"; do

      JOB_NAME="W2_Pipeline_${EMBEDDER}_V-${NUM_VISITS}_S-${SAMPLE_SIZE}"

      echo "--------------------------------------------------"
      echo "Submitting Full Pipeline Job: $JOB_NAME"
      echo "--------------------------------------------------"

      export EMBEDDER_TYPE_ENV="$EMBEDDER"
      export NUM_VISITS_ENV="$NUM_VISITS"
      export SAMPLE_SIZE_ENV="$SAMPLE_SIZE"

      # --- ✨ THE MAGIC PART! ✨ ---
      # We add --parsable to get the Job ID back! So clever!
      JOB_ID=$(sbatch \
        --parsable \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}.out.txt" \
        --error="${LOG_DIR}/${JOB_NAME}.err.txt" \
        --gres=gpu:"$GPU_COUNT" \
        --cpus-per-task="$CPU_COUNT" \
        --export=ALL \
        slurm_jobs/world_2_neighbor_analysis/world2_pipeline_template.ssub)
      
      echo "--> Job submitted with ID: $JOB_ID"
      # And now we write the secret cancellation command to our emergency script!
      echo "scancel $JOB_ID # $JOB_NAME" >> "$CANCEL_SCRIPT_PATH"

    done
  done
done

echo -e "\n--- ✅ All magnificent pipeline jobs launched! ---"
echo "🚨 To cancel ALL submitted jobs from this run, use: bash ${CANCEL_SCRIPT_PATH}"