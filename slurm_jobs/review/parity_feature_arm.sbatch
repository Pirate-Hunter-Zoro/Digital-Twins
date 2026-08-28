#!/bin/bash
#SBATCH --job-name=parity_feature
#SBATCH --partition=c3
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/review/logs/parity_feature_out.txt
#SBATCH -e slurm_jobs/review/logs/parity_feature_err.txt

# =============================================================================
# Review round of 2026-08-28, representation-parity item -- the CPU half.
#
# Refits the feature arm with the three vital-sign columns and a missingness
# indicator per vital block, then compares it to the published feature arm with
# a paired bootstrap on the same held-out patients.
#
# No GPU: this is two classifiers on 64 columns. The narrative half of the
# parity check needs a re-render and a re-embed and lives in
# parity_narrative_arms.sbatch.
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

# GridSearchCV already parallelises over 16 processes; leaving BLAS threaded on
# top of that oversubscribes the allocation.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m scripts.pipeline.review.parity.run_feature_arm

LOCAL_RESULTS_DIR="results/review/parity"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av --exclude '*.joblib' --exclude 'narratives/' --exclude '*.db' \
  "${ARTIFACTS_DIR}/review/parity/" "${LOCAL_RESULTS_DIR}/"

echo "Done."
