#!/bin/bash

set -e
echo "--- 🚀 Launching the FULL World 4 Gauntlet Workflow! (Both Metrics!) 🚀 ---"

# --- Get our bearings! ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- Step 1: Submit the Term Pair Generation Job ---
echo "STEP 1: Submitting the term pair generation job."
PAIR_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/generate_term_pairs.ssub")
echo "--> Term pair generation job submitted with ID: $PAIR_JOB_ID"

# --- Step 2: Launch all the calculation jobs (they will wait for Step 1) ---
echo "STEP 2: Submitting all similarity calculation launchers (will wait for job $PAIR_JOB_ID)"
MODERN_SIM_ID=$(sbatch --parsable --job-name="launch_modern_sim" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_modern_sim_out.txt" --error="${LOG_DIR}/launch_modern_sim_err.txt" "${SCRIPT_DIR}/launch_semantic_similarity_grid.sh")
LEGACY_SIM_ID=$(sbatch --parsable --job-name="launch_legacy_sim" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_legacy_sim_out.txt" --error="${LOG_DIR}/launch_legacy_sim_err.txt" "${SCRIPT_DIR}/launch_legacy_similarity_grid.sh")
MODERN_NORM_ID=$(sbatch --parsable --job-name="launch_modern_norm" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_modern_norm_out.txt" --error="${LOG_DIR}/launch_modern_norm_err.txt" "${SCRIPT_DIR}/launch_normalized_similarity_grid.sh")
LEGACY_NORM_ID=$(sbatch --parsable --job-name="launch_legacy_norm" --dependency=afterok:$PAIR_JOB_ID --output="${LOG_DIR}/launch_legacy_norm_out.txt" --error="${LOG_DIR}/launch_legacy_norm_err.txt" "${SCRIPT_DIR}/launch_legacy_normalized_similarity_grid.sh")

echo "--> Submitted calculation launchers with IDs: $MODERN_SIM_ID, $LEGACY_SIM_ID, $MODERN_NORM_ID, $LEGACY_NORM_ID"

# --- Step 3: Launch the plotting jobs (they will wait for ALL calculations) ---
echo "STEP 3: Submitting plotting jobs (will wait for all calculation launchers to finish)"

# The original plotting job
sbatch --job-name="plot_cosine_sim" --dependency=afterok:$MODERN_SIM_ID:$LEGACY_SIM_ID --output="${LOG_DIR}/plot_cosine_out.txt" --error="${LOG_DIR}/plot_cosine_err.txt" "${SCRIPT_DIR}/plot_similarity_distributions.ssub"
# The NEW normalized plotting job with the FIX!
sbatch --job-name="plot_normalized_sim" --dependency=afterok:$MODERN_NORM_ID:$LEGACY_NORM_ID --output="${LOG_DIR}/plot_normalized_out.txt" --error="${LOG_DIR}/plot_normalized_err.txt" "${SCRIPT_DIR}/plot_normalized_distributions.ssub"

echo ""
echo "✅ Entire magnificent workflow submitted! The dominoes are set!"
echo "Monitor progress with 'squeue -u your_username'"