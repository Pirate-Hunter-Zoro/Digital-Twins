#!/bin/bash
#SBATCH --job-name=ml_only
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/quick_runs/logs/analysis_only_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/analysis_only_err.txt
#SBATCH --gres=gpu:1

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
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# Log the allocated GPU so a later CUDA OOM is explainable from this log alone
echo "Running on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

python -m scripts.pipeline.predictions.classical_ml
python -m scripts.pipeline.predictions.feature_importance
python -m scripts.pipeline.predictions.plot_discrimination_forest
python -m scripts.pipeline.predictions.ablation_runner

# Copy all files from the RESULTS_DIR to the local results/ directory for easier access
LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"