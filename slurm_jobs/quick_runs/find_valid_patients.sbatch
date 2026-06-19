#!/bin/bash
#SBATCH --job-name=find_valid_patients
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o slurm_jobs/quick_runs/logs/find_valid_patients_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/find_valid_patients_err.txt

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load live vLLM endpoint 
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found. The server job must run first." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

srun --mem=0 python -m scripts.data_loading.fit_to_anchor