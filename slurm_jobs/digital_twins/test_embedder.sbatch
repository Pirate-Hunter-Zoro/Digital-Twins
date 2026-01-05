#!/bin/bash
#SBATCH --job-name=pipeline
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -o slurm_jobs/digital_twins/logs/test_embedder_out.txt
#SBATCH -e slurm_jobs/digital_twins/logs/test_embedder_err.txt

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=ehr_env
N_THREADS="${SLURM_CPUS_PER_TASK}"

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

# Threading sanity
export OMP_NUM_THREADS="${N_THREADS}"

srun python -m scripts.models.string_embedder