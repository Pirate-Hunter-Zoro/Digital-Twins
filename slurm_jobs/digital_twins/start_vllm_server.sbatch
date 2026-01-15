#!/bin/bash
#SBATCH --job-name=vllm-medgemma
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/digital_twins/logs/vllm_out.txt
#SBATCH -e slurm_jobs/digital_twins/logs/vllm_err.txt
#SBATCH --gres=gpu:2
  
GPUS=2                      
CONDA_ENV=ehr_env
TP_SIZE=$GPUS  
PP_SIZE=1                   
PORT=8000
DTYPE=bfloat16

# start clean
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
conda activate ehr_env

# Ensure proper gpu availability
nvidia-smi -L

# Modify the .env to contain the correct VLLM URL so other jobs send requests to the server
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
HEAD_NODE=$(hostname)
sed -i '/^VLLM_URL=/d' "${ENV_FILE}"
echo "VLLM_URL=http://${HEAD_NODE}:${PORT}" >> "${ENV_FILE}"

# Source file to load its variables (like VLLM_MODEL_PATH and MAX_MODEL_LEN)
if [[ -f "${ENV_FILE}" ]]; then
    set -a; source "${ENV_FILE}"; set +a
else
    echo "ERROR: .env file missing at ${ENV_FILE}"
    exit 1
fi

# ---- Start the server ----
srun python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL_PATH}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests