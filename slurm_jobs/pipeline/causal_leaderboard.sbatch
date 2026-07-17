#!/bin/bash
#SBATCH --job-name=causal_leaderboard
#SBATCH --partition=c3_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/pipeline/logs/causal_leaderboard_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/causal_leaderboard_err.txt

# afterok reduce for causal_forest_sweep.sbatch: globs every results_*.json the
# sweep wrote, BH-corrects across the passed contrasts, and writes the single
# leaderboard.csv. Submitted with --dependency=afterok by the sweep's first
# array task, so it only ever runs once all contrasts succeeded -> the glob is
# guaranteed non-empty (build_leaderboard has no empty-glob guard by design).

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/causal_forest_env

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load .env (ARTIFACTS_DIR)
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

python -m scripts.pipeline.causal.build_leaderboard
