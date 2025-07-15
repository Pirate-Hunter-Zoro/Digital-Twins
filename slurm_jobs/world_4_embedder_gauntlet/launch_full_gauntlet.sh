#!/bin/bash

# This is the ONE SCRIPT to rule them all!
set -e
echo "--- 🚀 Launching the FULL World 4 Gauntlet Workflow! 🚀 ---"

# --- ✨ The One True Source of Hardware Requests! ✨ ---
GPU_COUNT=1
CPU_COUNT=18

# --- Get our bearings! ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/logs"
CANCEL_SCRIPT_PATH="${SCRIPT_DIR}/cancel_gauntlet.sh" # Our new emergency stop button!
mkdir -p "$LOG_DIR"

# --- Create the cancel script from scratch! ---
echo "#!/bin/bash" > "$CANCEL_SCRIPT_PATH"
echo "echo '--- 🛑 Sending stop signal to all gauntlet jobs! 🛑 ---'" >> "$CANCEL_SCRIPT_PATH"
chmod +x "$CANCEL_SCRIPT_PATH"
echo "✅ Created new cancellation script at ${CANCEL_SCRIPT_PATH}"

# --- Step 1: Submit the Term Pair Generation Job ---
echo "STEP 1: Submitting the term pair generation job with ${GPU_COUNT} GPUs."
# This job gets its hardware requests directly!
PAIR_JOB_ID=$(sbatch --parsable \
                   --gres=gpu:$GPU_COUNT \
                   --cpus-per-task=$CPU_COUNT \
                   "${SCRIPT_DIR}/generate_term_pairs.ssub")
echo "--> Term pair generation job submitted with ID: $PAIR_JOB_ID"
echo "scancel $PAIR_JOB_ID # Term Pair Generation" >> "$CANCEL_SCRIPT_PATH"


# --- Step 2: Launch all the calculation jobs (they will wait for Step 1) ---
echo "STEP 2: Submitting all similarity calculation launchers (will wait for job $PAIR_JOB_ID)"

# Export the hardware requests for the sub-launchers to use!
export GPU_COUNT_ENV=$GPU_COUNT
export CPU_COUNT_ENV=$CPU_COUNT

# Submit the original cosine similarity jobs
MODERN_SIM_ID=$(sbatch --parsable --job-name="launch_modern_sim" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_modern_sim_out.txt" --error="${LOG_DIR}/launch_modern_sim_err.txt" --export=ALL "${SCRIPT_DIR}/launch_semantic_similarity_grid.sh")
echo "scancel $MODERN_SIM_ID # Modern Similarity Launcher" >> "$CANCEL_SCRIPT_PATH"

LEGACY_SIM_ID=$(sbatch --parsable --job-name="launch_legacy_sim" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_legacy_sim_out.txt" --error="${LOG_DIR}/launch_legacy_sim_err.txt" --export=ALL "${SCRIPT_DIR}/launch_legacy_similarity_grid.sh")
echo "scancel $LEGACY_SIM_ID # Legacy Similarity Launcher" >> "$CANCEL_SCRIPT_PATH"

# Submit the NEW normalized similarity jobs
MODERN_NORM_ID=$(sbatch --parsable --job-name="launch_modern_norm" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_modern_norm_out.txt" --error="${LOG_DIR}/launch_modern_norm_err.txt" --export=ALL "${SCRIPT_DIR}/launch_normalized_similarity_grid.sh")
echo "scancel $MODERN_NORM_ID # Modern Normalized Launcher" >> "$CANCEL_SCRIPT_PATH"

LEGACY_NORM_ID=$(sbatch --parsable --job-name="launch_legacy_norm" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_legacy_norm_out.txt" --error="${LOG_DIR}/launch_legacy_norm_err.txt" --export=ALL "${SCRIPT_DIR}/launch_legacy_normalized_similarity_grid.sh")
echo "scancel $LEGACY_NORM_ID # Legacy Normalized Launcher" >> "$CANCEL_SCRIPT_PATH"

echo "--> Submitted calculation launchers with IDs: $MODERN_SIM_ID, $LEGACY_SIM_ID, $MODERN_NORM_ID, $LEGACY_NORM_ID"

# --- Step 3: Launch the plotting jobs (they will wait for ALL calculations) ---
echo "STEP 3: Submitting plotting jobs (will wait for all calculation launchers to finish)"

# The original plotting job
PLOT_COSINE_ID=$(sbatch --parsable --job-name="plot_cosine_sim" --dependency=afterok:$MODERN_SIM_ID:$LEGACY_SIM_ID --output="${LOG_DIR}/plot_cosine_out.txt" --error="${LOG_DIR}/plot_cosine_err.txt" "${SCRIPT_DIR}/plot_similarity_distributions.ssub")
echo "scancel $PLOT_COSINE_ID # Cosine Plotting Job" >> "$CANCEL_SCRIPT_PATH"

# The NEW normalized plotting job
PLOT_NORM_ID=$(sbatch --parsable --job-name="plot_normalized_sim" --dependency=afterok:$MODERN_NORM_ID:$LEGACY_NORM_ID --output="${LOG_DIR}/plot_normalized_out.txt" --error="${LOG_DIR}/plot_normalized_err.txt" "${SCRIPT_DIR}/plot_normalized_distributions.ssub")
echo "scancel $PLOT_NORM_ID # Normalized Plotting Job" >> "$CANCEL_SCRIPT_PATH"


echo ""
echo "✅ Entire magnificent workflow submitted! The dominoes are set!"
echo "🚨 To cancel ALL submitted jobs, run: bash ${CANCEL_SCRIPT_PATH}"