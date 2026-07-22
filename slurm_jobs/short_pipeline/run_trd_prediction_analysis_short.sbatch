#!/bin/bash
#SBATCH --job-name=short_trd_prediction_analysis
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/short_pipeline/logs/trd_prediction_analysis_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/trd_prediction_analysis_err.txt

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

python -m scripts.pipeline.embeddings.embedding_audit

# LLM similarity audit only applies when judging ran (it did not here).
if [[ "$COMPUTE_LLM_SIMILARITY" == 1 ]]; then
  python -m scripts.pipeline.neighbors.llm_similarity_audit
fi

python -m scripts.pipeline.predictions.analyze_trd_prediction

# classical_ml (the grid-search wall-hitter), feature_importance, and the
# ablation report were SPLIT OUT into run_classical_ml_short.sbatch (2026-07-22)
# so this light neighbor-analysis arm no longer shares a wall with the ~2.5h
# EMBEDDED GridSearchCV sweep. The SINGLE results rsync likewise moved to
# run_results_mirror_short.sbatch, the terminal job that waits on BOTH this job
# and the classical-ML job before copying a complete result set. See the
# orchestrator DAG.
