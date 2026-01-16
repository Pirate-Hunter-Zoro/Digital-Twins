#!/bin/bash
#SBATCH --job-name=trd_prediction_pipeline
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/digital_twins/logs/trd_prediction_pipeline_%a_out.txt
#SBATCH -e slurm_jobs/digital_twins/logs/trd_prediction_pipeline_%a_err.txt
#SBATCH --array=0-19

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=ehr_env

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load live vLLM endpoint 
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

python -m scripts.digital_twins.predict_trd