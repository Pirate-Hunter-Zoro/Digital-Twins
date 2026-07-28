#!/bin/bash
#SBATCH --job-name=trd_pred_shard1_resume
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=256G
#SBATCH -o slurm_jobs/pipeline/trd_prediction/logs/trd_prediction_pipeline_1_out.txt
#SBATCH -e slurm_jobs/pipeline/trd_prediction/logs/trd_prediction_pipeline_1_err.txt

# One-off resume of ONLY shard 1 of the 2-shard judging run. Shard 0 is already
# complete (judgements_0.db + neighbor_results_0.csv on disk) and is left
# untouched. Shard 1 resumes from judgements_1.db via the scorer cache.
#
# We force SLURM_ARRAY_TASK_ID/COUNT to 1-of-2 by hand: resubmitting as a real
# --array=1-1 would set SLURM_ARRAY_TASK_COUNT=1 and the chunk math in
# run_neighborhood_constructor would mis-slice the cohort (empty chunk).

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# Reproduce "task 1 of 2" exactly.
export SLURM_ARRAY_TASK_ID=1
export SLURM_ARRAY_TASK_COUNT=2

TASK_IDX="${SLURM_ARRAY_TASK_ID}"
ENDPOINT_FILE="${ENDPOINTS_DIR}/endpoint_${TASK_IDX}.url"

# Wait (up to 30 min) for the freshly-launched shard-1 vLLM server to publish
# its endpoint and answer /health before judging.
echo "Waiting for vLLM endpoint ${ENDPOINT_FILE} to become healthy..."
for _ in $(seq 1 180); do
  if [[ -s "${ENDPOINT_FILE}" ]]; then
    URL="$(cat "${ENDPOINT_FILE}")"
    if curl -sf "${URL}/health" >/dev/null 2>&1; then
      echo "vLLM server live at ${URL}"
      break
    fi
  fi
  sleep 10
done

if [[ ! -s "${ENDPOINT_FILE}" ]]; then
  echo "ERROR: endpoint file never appeared: ${ENDPOINT_FILE}" >&2
  exit 12
fi
export VLLM_URL="$(cat "${ENDPOINT_FILE}")"
if ! curl -sf "${VLLM_URL}/health" >/dev/null 2>&1; then
  echo "ERROR: vLLM server not healthy at ${VLLM_URL}" >&2
  exit 13
fi
echo "Task ${TASK_IDX} targeting vLLM server at ${VLLM_URL}"

python -m scripts.pipeline.predictions.run_neighborhood_constructor
