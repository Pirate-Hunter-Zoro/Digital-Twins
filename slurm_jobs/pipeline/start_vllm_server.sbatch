#!/bin/bash
#SBATCH --job-name=vllm-medgemma
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/logs/vllm_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/vllm_err.txt
#SBATCH --gres=gpu:3
  
GPUS=2                      
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
TP_SIZE=$GPUS  
PP_SIZE=1                   
PORT=8000
DTYPE=bfloat16

# start clean
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Ensure proper gpu availability
nvidia-smi -L

# Modify the .env to contain the correct VLLM URL so other jobs send requests to the server
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
HEAD_NODE=$(hostname)
sed -i '/^VLLM_URL=/d' "${ENV_FILE}" # Remove existing VLLM_URL line if present
echo "VLLM_URL=http://${HEAD_NODE}:${PORT}" >> "${ENV_FILE}" # Append the new VLLM_URL line

# Source file to load its variables (like VLLM_MODEL_PATH and MAX_MODEL_LEN)
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
else
    echo "ERROR: .env file missing at ${ENV_FILE}"
    exit 1
fi

# Pin to the GPUs with the most free memory (works around bare-metal squatters slurm cannot see)
PICKED_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | sort -t, -k2 -n -r | head -2 | awk -F, '{print $1}' | paste -sd,)
export CUDA_VISIBLE_DEVICES="${PICKED_GPUS}"
echo "Pinning CUDA_VISIBLE_DEVICES to ${CUDA_VISIBLE_DEVICES} on $(hostname)"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# ---- Start the server ----
export VLLM_USE_V1=0
python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests