#!/bin/bash
#SBATCH --job-name=religion_feature
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/review/logs/religion_feature_out.txt
#SBATCH -e slurm_jobs/review/logs/religion_feature_err.txt

# =============================================================================
# Review round of 2026-09-02, religion sensitivity item -- the CPU half.
#
# Refits the feature arm with the Religion column removed and compares it to the
# published feature arm with a paired bootstrap on the same held-out patients.
# The field is recorded for 70.6% of the cohort and the one-hot encoder gives its
# unrecorded value a level of its own, which carries an age gradient -- 43.8%
# unrecorded at 18-29 against 17.5% at 65 or older -- so the model can read the
# field's absence as an age proxy. Removing the column takes both the value and
# its absence away, which is the question the review asked and a different one
# from the ablation, which permutes.
#
# No GPU: this is two classifiers on 91 columns. The narrative half needs a
# re-render and a re-embed and lives in religion_narrative_embed.sbatch.
#
# PARTITION: c3_short rather than c3, for the reason recorded in
# parity_feature_arm.sbatch -- same nodes, higher priority tier, 9-hour wall.
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
# top of that oversubscribes the allocation. Ninety-one columns make BLAS
# threading pure overhead anyway.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m scripts.pipeline.review.religion.run_feature_arm
python -m scripts.pipeline.review.religion.summarize

LOCAL_RESULTS_DIR="results/review/religion"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av --exclude '*.joblib' --exclude 'narratives/' --exclude '*.db' \
  "${ARTIFACTS_DIR}/review/religion/" "${LOCAL_RESULTS_DIR}/"

echo "Done."
