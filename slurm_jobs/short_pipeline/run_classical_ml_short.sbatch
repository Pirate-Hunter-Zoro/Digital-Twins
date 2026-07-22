#!/bin/bash
#SBATCH --job-name=short_classical_ml
#SBATCH --partition=c3_short
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/short_pipeline/logs/classical_ml_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/classical_ml_err.txt

# =============================================================================
# SHORT pipeline -- classical-ML arm (split out of the analysis job 2026-07-22).
#
# The EMBEDDED GridSearchCV sweep is the only step that threatened the c3_short
# wall (logistic_regression alone took ~2.5h on the 2560-dim 4B embeddings, and
# the old shared 8h analysis job died mid-gradient_boosting), so it now owns a
# dedicated 9h job -- the c3_short MaxTime -- instead of sharing a wall with the
# neighbor analysis. Fitted searchers are joblib-cached to
# RESULTS_DIR/trained_models with SCRUB_TRAINED_MODELS=0, so a re-run loads any
# classifier that already finished and resumes at the first that did not.
#
# Steps: classical_ml -> feature_importance -> ablation_runner (guarded). No
# rsync here; the single results mirror is run_results_mirror_short.sbatch, the
# terminal job that waits on BOTH this job and the neighbor-analysis job.
# =============================================================================

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
N_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# ---- Config: real .env, then the short-pipeline overrides ----
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

python -m scripts.pipeline.predictions.classical_ml
python -m scripts.pipeline.predictions.feature_importance

# The ablation report consumes per-spec ablated embeddings; only run it when the
# ablation embeddings were actually built (see RUN_ABLATIONS in short_overrides.env).
if [[ "${RUN_ABLATIONS}" == 1 ]]; then
  python -m scripts.pipeline.predictions.ablation_runner
fi
