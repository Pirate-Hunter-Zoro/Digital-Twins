#!/bin/bash
#SBATCH --job-name=subgroup_perf
#SBATCH --partition=c3
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o slurm_jobs/review/logs/subgroup_perf_out.txt
#SBATCH -e slurm_jobs/review/logs/subgroup_perf_err.txt

# =============================================================================
# Review round of 2026-08-28, subgroup-performance item.
#
# Discrimination and calibration for the published models inside every stratum
# the cohort can estimate: six sociodemographic families (sex, race, age band,
# marital status, smoking status, religion) and two clinical ones (MDD severity,
# MDD recurrence). Preferred language is absent because 98.9% of the cohort
# prefers English, leaving one estimable level and therefore no contrast.
#
# THREE ARMS, not two. The embedded and feature-vector classifiers come from
# test_predictions_{EMBEDDED,FEATURE}.parquet; the neighbour-weighted predictor
# comes from summary_predictions.csv, whose sixteen retrieval-scheme x
# weighting-strategy pairs are each scored per group. Only nearest retrieval is
# CONTRASTED across groups -- whether a deliberately uninformative retrieval
# scheme is evenly uninformative across subgroups is not a fairness question,
# and the three negative-control schemes would triple the multiplicity burden on
# the contrasts that are.
#
# NOTHING IS REFIT and no GPU is needed. Subsetting a per-patient prediction
# vector by group is arithmetically identical to scoring that model on the
# group, so the published numbers cannot drift. Refitting per subgroup would
# answer a different question -- how a model trained only on one group predicts
# that group -- and would break comparability with everything published.
#
# The wall is bootstrap, not modelling: roughly 290 between-group contrasts and
# 750 group-model cells, each drawing N_BOOTSTRAP resamples and scoring an AUC
# per draw. That is over a million roc_auc_score calls, which is why this has
# cores at all -- SUBGROUP_JOBS fans the two loops over them with joblib. Every
# unit re-seeds from SEED, so the numbers do not depend on how the work was
# split; the parallelism buys wall-clock and nothing else.
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

# joblib fans out one process per core; leaving BLAS threaded on top of that
# oversubscribes the allocation.
export SUBGROUP_JOBS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m scripts.pipeline.review.subgroups.run_subgroups

LOCAL_RESULTS_DIR="results/review/subgroups"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${ARTIFACTS_DIR}/review/subgroups/" "${LOCAL_RESULTS_DIR}/"

echo "Done."
