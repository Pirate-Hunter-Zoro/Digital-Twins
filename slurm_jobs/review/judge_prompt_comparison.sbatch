#!/bin/bash
#SBATCH --job-name=judge_prompt_cmp
#SBATCH --partition=c3_accel
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH -o slurm_jobs/review/logs/judge_prompt_cmp_out.txt
#SBATCH -e slurm_jobs/review/logs/judge_prompt_cmp_err.txt

# =============================================================================
# Review round of 2026-08-28, judge-rubric item.
#
# Re-judges a sample of already-cached pairs under the phenotype-free rubric and
# correlates the new overall_similarity against the cached one.
#
# Self-contained on purpose: it starts ONE vLLM server in the background on its
# own allocation, waits for /health, runs the comparison, and kills the server
# on the way out. The pipeline's usual arrangement -- a separate 7-day server
# array plus an orchestrator that polls endpoints -- exists so several judging
# tasks can share servers; here there is one consumer and one server, and the
# review annotation asks for exactly one LLM instance, so a background process
# beats a job that outlives its only client.
#
# The judged sample is small (5,000 pairs against 1.71M in the cache), so the
# wall is dominated by model load, not by generation.
# =============================================================================

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
N_PAIRS="${N_PAIRS:-5000}"
PORT=8100
GPUS=2
DTYPE=bfloat16

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

cd "${SLURM_SUBMIT_DIR}"
nvidia-smi -L

# ---- One server, on a port the pipeline's servers do not use (8000/8001) ----
echo "Starting vLLM on port ${PORT}..."
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_PATH}" \
  --tensor-parallel-size "${GPUS}" \
  --pipeline-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs "${LLM_MAX_CONCURRENCY}" &
VLLM_PID=$!

# Kill the server however this script exits, including on the 'set -e' path.
cleanup() {
  echo "Stopping vLLM (pid ${VLLM_PID})..."
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export VLLM_URL="http://127.0.0.1:${PORT}"

echo "Waiting for ${VLLM_URL}/health ..."
for _ in $(seq 1 180); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming healthy." >&2
    exit 12
  fi
  if curl -sf "${VLLM_URL}/health" >/dev/null; then
    echo "  --> server live."
    break
  fi
  sleep 10
done
curl -sf "${VLLM_URL}/health" >/dev/null || { echo "ERROR: server never became healthy." >&2; exit 13; }

python -m scripts.pipeline.review.judge_prompt.run_comparison "${N_PAIRS}"

# ---- Mirror into the repo results tree, same convention as the pipeline ----
LOCAL_RESULTS_DIR="results/review/judge_prompt"
mkdir -p "${LOCAL_RESULTS_DIR}"
# The .db is the analysis' own cache and is large; the mirror carries the
# findings, not the cache.
rsync -av --exclude '*.db' "${ARTIFACTS_DIR}/review/judge_prompt/" "${LOCAL_RESULTS_DIR}/"

echo "Done."
