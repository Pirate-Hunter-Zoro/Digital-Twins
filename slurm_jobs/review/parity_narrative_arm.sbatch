#!/bin/bash
#SBATCH --job-name=parity_narr
#SBATCH --partition=c3_accel
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/review/logs/parity_narr_%a_out.txt
#SBATCH -e slurm_jobs/review/logs/parity_narr_%a_err.txt
#SBATCH --array=0-1
#
# Review round of 2026-08-28, representation-parity item -- the GPU half.
#
# One array index per narrative arm:
#   0  narrative_control  re-render with NO content change
#   1  narrative_parity   re-render with pre_anchor_history_days added
#
# The control arm is not padding. render_narrative used to emit its flag lists in
# set-iteration order, which varies per process, so the published narratives are
# not byte-reproducible; the order is sorted now, which fixes future renders but
# does not reproduce what was embedded. Arm 0 measures what a re-render alone
# costs, so arm 1's delta can be attributed to the added field rather than to the
# reshuffle. It also answers a question the paper never had an answer for: how
# stable the embedded arm is to a rebuild of its own inputs.
#
# Four stages per arm, all writing under ARTIFACTS_DIR/review/parity/<arm>/:
#   render   42,579 baseline narratives, CPU, no ablation specs
#   embed    the same encoder as the published run, GPU, ~3h
#   neighbours  cosine retrieval only -- COMPUTE_LLM_SIMILARITY=0 drops the LLM
#               and COMBINED weighting strategies, leaving UNIFORM and COSINE,
#               which is the "KNN without judgement weighting" the review asked
#               for. No vLLM server is involved.
#   fit      logistic regression and XGBoost, then the paired comparison

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

cd "${SLURM_SUBMIT_DIR}"

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${IDX}" == 0 ]]; then
  ARM="narrative_control"
  export NARRATIVE_INCLUDE_HISTORY_LENGTH=0
else
  ARM="narrative_parity"
  export NARRATIVE_INCLUDE_HISTORY_LENGTH=1
fi
echo "Parity arm ${IDX}: ${ARM} (history length ${NARRATIVE_INCLUDE_HISTORY_LENGTH})"

ARM_DIR="${ARTIFACTS_DIR}/review/parity/${ARM}"
mkdir -p "${ARM_DIR}"

# Every path the pipeline stages read or write is redirected into this arm, so no
# published artifact is touched: not the narratives, not the embedding database,
# not the judgement cache (the Scorer still opens one even with judging off), and
# not RESULTS_DIR, which is where the neighbour stage drops its CSVs and plots.
export NARRATIVES_DIR="${ARM_DIR}/narratives"
export EMBEDDINGS_DIR="${ARM_DIR}/${EMBEDDER_MODEL_NAME}"
export JUDGEMENTS_DIR="${ARM_DIR}/judgements"
export RESULTS_DIR="${ARM_DIR}"
export COMPUTE_LLM_SIMILARITY=0
export NEIGHBOR_FARTHEST=0
export NEIGHBOR_SUBSAMPLE=0
export SCRUB_EMBEDDINGS=0
mkdir -p "${NARRATIVES_DIR}" "${EMBEDDINGS_DIR}" "${JUDGEMENTS_DIR}"

echo "=== Stage 1/4: render narratives ==="
python -m scripts.pipeline.review.parity.render_narratives "${ARM}"

echo "=== Stage 2/4: embed narratives ==="
python -m scripts.pipeline.embeddings.forge_embeddings

echo "=== Stage 3/4: neighbours (cosine only) ==="
# The neighbour constructor shards over a SLURM array; this is a single-process
# run, so it is told it is task 0 of 1.
SLURM_ARRAY_TASK_ID=0 SLURM_ARRAY_TASK_COUNT=1 \
  python -m scripts.pipeline.predictions.run_neighborhood_constructor
python -m scripts.pipeline.predictions.analyze_trd_prediction

echo "=== Stage 4/4: fit classifiers and compare ==="
# RESULTS_DIR has to point back at the PUBLISHED results for this stage, since
# that is where the published test_predictions parquet lives.
RESULTS_DIR="${ARTIFACTS_DIR}/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}" \
  python -m scripts.pipeline.review.parity.run_embedded_arm "${ARM}"

LOCAL_RESULTS_DIR="results/review/parity"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av --exclude '*.joblib' --exclude 'narratives/' --exclude '*.db' \
  --exclude 'judgements/' --exclude 'neighbor_results_*.csv' \
  "${ARTIFACTS_DIR}/review/parity/" "${LOCAL_RESULTS_DIR}/"

echo "Done: ${ARM}."
