#!/bin/bash
#SBATCH --job-name=causal_forest_orchestrator
#SBATCH --partition=c3_short
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/pipeline/logs/causal_forest_orchestrator_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/causal_forest_orchestrator_err.txt

# Causal orchestrator: submits the pairwise causal-forest sweep (array 0-2, one
# contrast per task) and chains the leaderboard reduce to run afterok on the
# WHOLE array, so build_leaderboard fires exactly once and only after all three
# contrasts succeed. Mirrors the prediction orchestrator's parsable-submit +
# afterok-dependency pattern. Submit THIS, not the sweep/leaderboard directly.

# Build a cancel list so a single script tears the whole chain down.
CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/pipeline/cancel_causal.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing causal forest sweep + leaderboard...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"

echo "Submitting causal forest sweep (array 0-2)..."
SWEEP_JOB_ID=$(sbatch --parsable slurm_jobs/pipeline/causal_forest_sweep.sbatch)
echo "scancel ${SWEEP_JOB_ID} # Causal Forest Sweep" >> "${CANCEL_SCRIPT}"

# afterok on the bare array job id is satisfied only when EVERY array task
# succeeds; if any contrast fails, SLURM cancels the queued leaderboard.
echo "Chaining leaderboard reduce (afterok on the full sweep array)..."
LEADERBOARD_JOB_ID=$(sbatch --parsable --dependency=afterok:${SWEEP_JOB_ID} \
  slurm_jobs/pipeline/causal_leaderboard.sbatch)
echo "scancel ${LEADERBOARD_JOB_ID} # Causal Leaderboard Reduce" >> "${CANCEL_SCRIPT}"

echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"

echo "Sweep=${SWEEP_JOB_ID} Leaderboard=${LEADERBOARD_JOB_ID}. Done."
