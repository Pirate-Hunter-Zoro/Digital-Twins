#!/bin/bash
#SBATCH --job-name=holdout_repr
#SBATCH --partition=c3_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/review/logs/holdout_repr_out.txt
#SBATCH -e slurm_jobs/review/logs/holdout_repr_err.txt

# =============================================================================
# Review round of 2026-08-28, held-out-representativeness item.
#
# Tabulates the held-out partition against the training partition, every column
# and level, with the standardized mean difference between them, plus a curated
# table matching the manuscript's cohort table row for row.
#
# This is a minutes-long job on one parquet read and it could be run in-process.
# It gets an sbatch anyway, for the same reason the other three review analyses
# have one: the mirror into results/ is part of the analysis, and an artifact
# copied by hand is an artifact nobody can reproduce. Small jobs that write
# published tables belong in the batch system with the large ones.
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

python -m scripts.pipeline.review.holdout_representativeness

LOCAL_RESULTS_DIR="results/review/holdout_representativeness"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${ARTIFACTS_DIR}/review/holdout_representativeness/" "${LOCAL_RESULTS_DIR}/"

echo "Done."
