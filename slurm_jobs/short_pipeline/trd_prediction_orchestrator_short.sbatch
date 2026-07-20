#!/bin/bash
#SBATCH --job-name=trd_short_orchestrator
#SBATCH --partition=c3_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/short_pipeline/logs/orchestrator_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/orchestrator_err.txt

# =============================================================================
# SHORT pipeline orchestrator (c3_short, 9h wall, LLM judging OFF).
#
# Chain:  embedding  ->  prediction (array 0-0)  ->  analysis
#
# The JSON build is SKIPPED on purpose: the sliced patient JSONs are model-
# independent and already exist under ANALYSIS_DIR from the main run, and
# rebuilding them is what hit the Isilon quota before. The embedding job
# re-derives narratives/vectors/embeddings into the isolated artifacts_short
# sandbox (see short_overrides.env), so the live run is never touched.
#
# LLM judging is intentionally unsupported here (it cannot clear a 9h wall).
# =============================================================================

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/cancel_short_pipeline.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing SHORT prediction pipeline...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"
echo "scancel ${SLURM_JOB_ID} # Orchestrator Job" >> "${CANCEL_SCRIPT}"

echo "Launching the embedding job (rebuilds artifacts_short from existing sliced JSONs)..."
EMBEDDING_JOB_ID=$(sbatch --parsable slurm_jobs/short_pipeline/run_embedding_pipeline_short.sbatch)
echo "scancel ${EMBEDDING_JOB_ID} # Embedding Job" >> "${CANCEL_SCRIPT}"

echo "Launching the prediction pipeline (array 0-0, no vLLM)..."
PREDICTION_JOB_ID=$(sbatch --parsable --array=0-0 --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/short_pipeline/run_trd_prediction_pipeline_short.sbatch)
echo "scancel ${PREDICTION_JOB_ID} # Prediction Pipeline" >> "${CANCEL_SCRIPT}"

echo "Launching the analysis pipeline (runs once prediction completes)..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$PREDICTION_JOB_ID slurm_jobs/short_pipeline/run_trd_prediction_analysis_short.sbatch)
echo "scancel ${ANALYSIS_JOB_ID} # Analysis Pipeline" >> "${CANCEL_SCRIPT}"

echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"
echo "Done. Cancel the whole chain with: bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh"
