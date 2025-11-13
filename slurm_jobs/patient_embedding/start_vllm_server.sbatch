#!/bin/bash
#SBATCH --job-name=vllm-medgemma
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/patient_embedding/logs/vllm-medgemma_out.txt
#SBATCH -e slurm_jobs/patient_embedding/logs/vllm-medgemma_err.txt
#SBATCH --gres=gpu:2
  
GPUS=2                      
CPUS=32      
CONDA_ENV=venv
MODEL_ID=/media/studies/ehr_study/data-EHR-prepped/Mikey-Digital-Twins/models/google_medgemma-27b-text-it
TP_SIZE=2  
PP_SIZE=1                   
PORT=8000
MAX_MODEL_LEN=131072
DTYPE=bfloat16

export OMP_NUM_THREADS="${CPUS}"

# start clean
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
conda activate venv

# Ensure proper gpu availability
nvidia-smi -L

# ---- Start the server ----
srun python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ID}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size "${PP_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --port "${PORT}" \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests