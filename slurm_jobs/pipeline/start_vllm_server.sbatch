#!/bin/bash
#SBATCH --job-name=vllm-medgemma
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/logs/vllm_%a_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/vllm_%a_err.txt
#SBATCH --gres=gpu:2
#SBATCH --array=0-0

GPUS=2
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
TP_SIZE=$GPUS
PP_SIZE=1
BASE_PORT=8000
DTYPE=bfloat16

# start clean
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Ensure proper gpu availability
nvidia-smi -L

# Source .env for VLLM_MODEL_PATH, MAX_MODEL_LEN, LLM_MAX_CONCURRENCY, ENDPOINTS_DIR
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
else
    echo "ERROR: .env file missing at ${ENV_FILE}"
    exit 1
fi

# This server's index within the array (default 0 for a standalone run)
IDX="${SLURM_ARRAY_TASK_ID:-0}"

# Per-task port so co-located servers don't collide (8000, 8001, ...)
PORT=$(( BASE_PORT + IDX ))

# Publish this server's URL to an index-keyed endpoint file (replaces the old
# .env VLLM_URL rewrite, which two servers would clobber). Fall back to a
# default location if ENDPOINTS_DIR is not set in .env.
ENDPOINTS_DIR="${ENDPOINTS_DIR:-${SLURM_SUBMIT_DIR}/slurm_jobs/pipeline/endpoints}"
mkdir -p "${ENDPOINTS_DIR}"
HEAD_NODE=$(hostname)
ENDPOINT_FILE="${ENDPOINTS_DIR}/endpoint_${IDX}.url"
# Atomic write so a consumer never reads a half-written path
printf '%s\n' "http://${HEAD_NODE}:${PORT}" > "${ENDPOINT_FILE}.tmp"
mv "${ENDPOINT_FILE}.tmp" "${ENDPOINT_FILE}"
echo "Published endpoint ${IDX}: http://${HEAD_NODE}:${PORT} -> ${ENDPOINT_FILE}"

# ---- Start the server ----
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs "${LLM_MAX_CONCURRENCY}"
