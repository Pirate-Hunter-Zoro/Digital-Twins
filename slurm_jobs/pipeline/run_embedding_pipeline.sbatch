#!/bin/bash
#SBATCH --job-name=embedding_pipeline
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/logs/embedding_pipeline_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/embedding_pipeline_err.txt
#SBATCH --gres=gpu:2

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

# Pin to the GPU with the most free memory (works around bare-metal squatters slurm cannot see)
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -n -r | head -1 | awk -F, '{print $1}' | tr -d ' ')
export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
echo "Pinning to GPU ${CUDA_VISIBLE_DEVICES} on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

python -m scripts.digital_twins.narratives.forge_narratives
python -m scripts.digital_twins.vectors.forge_vectors
python -m scripts.digital_twins.embeddings.forge_embeddings
python -m scripts.digital_twins.predictions.ablation_runner