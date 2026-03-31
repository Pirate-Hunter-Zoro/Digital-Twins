#!/bin/bash
#SBATCH --job-name=deterministic_data
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -o slurm_jobs/quick_runs/logs/deterministic_data_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/deterministic_data_err.txt

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
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

python -m scripts.data_loading.build_jsons
python -m scripts.digital_twins.narratives.forge_narratives
python -m scripts.digital_twins.vectors.forge_vectors

# Copy all files from the RESULTS_DIR to the local test_data directory for easier access
LOCAL_RESULTS_DIR="test_data/"
rsync -av "${RESULTS_DIR}/vector_sanity_check.txt" "${LOCAL_RESULTS_DIR}/"