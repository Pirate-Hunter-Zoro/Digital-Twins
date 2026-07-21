#!/bin/bash
#SBATCH --job-name=causal_forest_orchestrator
#SBATCH --partition=c3_short
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/pipeline/causal/logs/causal_forest_orchestrator_out.txt
#SBATCH -e slurm_jobs/pipeline/causal/logs/causal_forest_orchestrator_err.txt

# Causal orchestrator: submits the pairwise causal-forest sweep (array 0-2, one
# contrast per task), chains the leaderboard reduce afterok on the WHOLE array,
# then chains the validation suite (sign-flip + transitivity) afterok on the
# leaderboard so it runs strictly last. The single results/ mirror lives in
# causal_validate.sbatch (the terminal job), so leaderboard no longer rsyncs.
# Submit THIS, not the sweep/leaderboard/validate directly.

# Build a cancel list so a single script tears the whole chain down.
CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/pipeline/causal/cancel_causal.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing causal forest sweep + leaderboard + validate...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"

echo "Submitting causal forest sweep (array 0-2)..."
SWEEP_JOB_ID=$(sbatch --parsable slurm_jobs/pipeline/causal/causal_forest_sweep.sbatch)
echo "scancel ${SWEEP_JOB_ID} # Causal Forest Sweep" >> "${CANCEL_SCRIPT}"

# afterok on the bare array job id is satisfied only when EVERY array task
# succeeds; if any contrast fails, SLURM cancels the queued leaderboard.
echo "Chaining leaderboard reduce (afterok on the full sweep array)..."
LEADERBOARD_JOB_ID=$(sbatch --parsable --dependency=afterok:${SWEEP_JOB_ID} \
  slurm_jobs/pipeline/causal/causal_leaderboard.sbatch)
echo "scancel ${LEADERBOARD_JOB_ID} # Causal Leaderboard Reduce" >> "${CANCEL_SCRIPT}"

# Chain the validation suite afterok on the leaderboard so it runs strictly last.
# Its rsync is the SINGLE mirror of the whole causal_pipeline dir (incl.
# leaderboard.csv + validation_report.json). validate reads the sweep's
# results_*.json, which exist by the time the leaderboard has succeeded.
echo "Chaining validation suite (afterok on the leaderboard)..."
VALIDATE_JOB_ID=$(sbatch --parsable --dependency=afterok:${LEADERBOARD_JOB_ID} \
  slurm_jobs/pipeline/causal/causal_validate.sbatch)
echo "scancel ${VALIDATE_JOB_ID} # Causal Validation Suite" >> "${CANCEL_SCRIPT}"

echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"

echo "Sweep=${SWEEP_JOB_ID} Leaderboard=${LEADERBOARD_JOB_ID} Validate=${VALIDATE_JOB_ID}. Done."
