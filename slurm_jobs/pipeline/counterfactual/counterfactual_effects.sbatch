#!/bin/bash
#SBATCH --job-name=counterfactual_effects
#SBATCH --partition=c3
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/counterfactual/logs/counterfactual_effects_%a_out.txt
#SBATCH -e slurm_jobs/pipeline/counterfactual/logs/counterfactual_effects_%a_err.txt
#SBATCH --array=0-2

# =============================================================================
# Paper 2 -- counterfactual (T-learner) treatment-selection pipeline.
#
# One pairwise contrast per array index (0,1,2 -> TREATMENT_REGISTRY entries,
# whose order is FROZEN; the array maps to it by position).
#
# The wall is dominated by the bootstrap, which refits all three models -- both
# arm risk models and the propensity model -- inside every draw, for both
# schemes: 2 x N_BOOTSTRAP full pipeline passes per contrast. Measured on
# compute301 the slowest contrast (snri_vs_ssri) runs a single pass in ~1.8s,
# so ~2000 passes over 24 cores lands well inside an hour; the 4h wall is slack,
# not an estimate. Note that the old frozen-vector bootstrap was instantaneous --
# if this job finishes in seconds, the refit is not happening.
#
# CPU-only. No --gres=gpu: this is logistic regression on 59 feature columns.
# =============================================================================

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
# NOT causal_forest_env. This package imports classical_ml.make_classifier to
# reuse Paper 1's ColumnTransformer, and that import fails under the causal env.
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load .env (SEED, ARTIFACTS_DIR, FEATURE_DATAFRAME_PATH, MDD_MED_DATE_CSV_PATH, ...)
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# Parallelism is joblib-over-draws, one process per core. Leaving BLAS threaded
# on top of that oversubscribes the allocation AND makes the intervals depend on
# the core count; core.py also pins the inner pool, and this is the belt to that
# pair of braces.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -m scripts.pipeline.counterfactual.run_one
