#!/bin/bash
#SBATCH --job-name=short_results_mirror
#SBATCH --partition=c3_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o slurm_jobs/short_pipeline/logs/results_mirror_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/results_mirror_err.txt

# =============================================================================
# SHORT pipeline -- terminal results mirror (split out 2026-07-22).
#
# The SINGLE rsync of the 4B encoder's RESULTS_DIR into the repo results/ tree,
# under the normal results/<encoder>/<judge> path -- same as the main pipeline.
# Chained afterok on BOTH the neighbor-analysis job and the classical-ML job, so
# it only ever copies a COMPLETE result set (neighbor diagnostics + classical ML
# + ablation report) exactly once. The 4B encoder name keeps this distinct from
# the live 8B run's mirror, so it is a first-class result, not a sandbox copy.
# =============================================================================

# Exit immediately if command fails
set -e

module purge

# ---- Config: real .env, then the short-pipeline overrides (for RESULTS_DIR,
#      EMBEDDER_MODEL_NAME, VLLM_MODEL_NAME) ----
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"

# Cancel the (already-finished) orchestrator/self entries.
bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh
