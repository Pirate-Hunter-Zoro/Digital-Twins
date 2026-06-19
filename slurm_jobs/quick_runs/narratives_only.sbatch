#!/bin/bash
#SBATCH --job-name=narratives_only
#SBATCH --partition=c3_short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/quick_runs/logs/narratives_only_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/narratives_only_err.txt

set -e

CONDA_ENV=ehr_env

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

export SCRUB_NARRATIVES=1

python -m scripts.embedder_investigation.narratives.forge_narratives
