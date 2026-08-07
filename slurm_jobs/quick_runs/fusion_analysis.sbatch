#!/bin/bash
#SBATCH --job-name=fusion_analysis
#SBATCH --partition=c3_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/quick_runs/logs/fusion_analysis_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/fusion_analysis_err.txt

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

# Load env variables 
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

python -m scripts.pipeline.predictions.fusion_analysis

TARGET_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}/"
mkdir -p "${TARGET_DIR}"
rsync -av "${RESULTS_DIR}/" "${TARGET_DIR}"
