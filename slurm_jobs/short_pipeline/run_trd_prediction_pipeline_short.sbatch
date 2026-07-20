#!/bin/bash
#SBATCH --job-name=short_trd_prediction_pipeline
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -o slurm_jobs/short_pipeline/logs/trd_prediction_pipeline_%a_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/trd_prediction_pipeline_%a_err.txt
#SBATCH --array=0-0

# NOTE: c3_short enforces MaxMemPerCPU=12000 MB. The main pipeline asks 256G on
# 2 CPUs (128 GB/CPU) which c3_short would reject, so this copy uses 32 CPUs
# (256G/32 = 8 GB/CPU). No GPU and no vLLM: cosine retrieval is pure NumPy.

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

# ---- Config: real .env, then the short-pipeline overrides ----
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

# LLM judging is forced off in short_overrides.env, so no endpoint is resolved
# and no vLLM server is contacted. (Guard kept for parity with the main job.)
if [[ "${COMPUTE_LLM_SIMILARITY}" == 1 ]]; then
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
