#!/bin/bash
#SBATCH --job-name=parity_narr_fit
#SBATCH --partition=c3_short
#SBATCH --time=08:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/review/logs/parity_narr_fit_%a_out.txt
#SBATCH -e slurm_jobs/review/logs/parity_narr_fit_%a_err.txt
#SBATCH --array=0-1

# =============================================================================
# Representation parity, narrative half -- CLASSIFIER FIT ONLY.
#
# parity_narrative_arm.sbatch runs four stages. Both of its array tasks reached
# the last one and were killed at the 12-hour wall part-way through the embedded
# logistic-regression grid search. Everything upstream survives on disk: 42,579
# narratives, a complete 42,579-row embeddings.db, and the cosine/uniform
# neighbour results. Only the fit is missing, and it is standalone, so this job
# picks up exactly there rather than re-rendering and re-embedding (~6h of GPU
# time) to reach the same point.
#
# WHY THE ORIGINAL RAN OUT. Two reasons, both fixed here.
#   Order. The submitted copy ran the neighbour stage third and the fit fourth,
#   so the stage the parity decision rests on got whatever time the cheaper
#   corroboration left it -- 5.9h of a 12h wall. The sbatch has since been
#   reordered; these arms were submitted before that.
#   Width. It asked for 16 CPUs. GridSearchCV runs 16 worker processes, so each
#   worker's BLAS had a single core to itself and threaded on top of a fully
#   subscribed allocation. The published run fit the same grid on the same
#   matrix in 4.9h with 48 CPUs and BLAS left threaded, which is what this job
#   asks for. Do not pin OMP_NUM_THREADS here: the feature arm pins it because
#   64 columns make BLAS threading pure overhead, and 4,096 columns do not.
#
# No GPU. This is two classifiers reading an embedding database.
# =============================================================================

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
else
  ARM="narrative_parity"
fi
echo "Parity fit, arm ${IDX}: ${ARM}"

ARM_DIR="${ARTIFACTS_DIR}/review/parity/${ARM}"
export NARRATIVES_DIR="${ARM_DIR}/narratives"
export EMBEDDINGS_DIR="${ARM_DIR}/${EMBEDDER_MODEL_NAME}"
export JUDGEMENTS_DIR="${ARM_DIR}/judgements"

# Preflight. load_embedded_matrix raises on a short database, but it raises
# after the split and the load, and a missing narrative directory would only
# surface later still. Fail here instead.
EMB_DB="${EMBEDDINGS_DIR}/embeddings.db"
if [[ ! -f "${EMB_DB}" ]]; then
  echo "ERROR: ${EMB_DB} not found; arm ${ARM} never finished embedding." >&2
  exit 12
fi

# RESULTS_DIR points at the PUBLISHED results, which is where the published
# test_predictions parquet the comparison is paired against lives.
RESULTS_DIR="${ARTIFACTS_DIR}/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}" \
  python -m scripts.pipeline.review.parity.run_embedded_arm "${ARM}"

LOCAL_RESULTS_DIR="results/review/parity"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av --exclude '*.joblib' --exclude 'narratives/' --exclude '*.db' \
  --exclude 'judgements/' --exclude 'neighbor_results_*.csv' \
  "${ARTIFACTS_DIR}/review/parity/" "${LOCAL_RESULTS_DIR}/"

echo "Done: ${ARM} fit."
