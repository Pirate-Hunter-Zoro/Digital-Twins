#!/bin/bash
#SBATCH --job-name=trd_prediction_pipeline
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH -o slurm_jobs/pipeline/logs/trd_prediction_pipeline_%a_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/trd_prediction_pipeline_%a_err.txt
#SBATCH --array=0-0

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load live vLLM endpoint 
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# Only resolve a vLLM endpoint when LLM judging is enabled. With similarity
# off, no servers are launched and no endpoint files exist, so this whole
# block is skipped.
if [[ "${COMPUTE_LLM_SIMILARITY}" == 1 ]]; then
  # Resolve this task's vLLM server from the index-keyed endpoint file the
  # server job published. Default the index to 0 so a standalone (non-array)
  # run works even with 'set -u' active.
  TASK_IDX="${SLURM_ARRAY_TASK_ID:-0}"
  ENDPOINT_FILE="${ENDPOINTS_DIR}/endpoint_${TASK_IDX}.url"

  if [[ ! -s "${ENDPOINT_FILE}" ]]; then
    echo "ERROR: endpoint file missing or empty: ${ENDPOINT_FILE}" >&2
    exit 12
  fi

  export VLLM_URL="$(cat "${ENDPOINT_FILE}")"
  echo "Task ${TASK_IDX} targeting vLLM server at ${VLLM_URL}"
fi

python -m scripts.pipeline.predictions.run_neighborhood_constructor