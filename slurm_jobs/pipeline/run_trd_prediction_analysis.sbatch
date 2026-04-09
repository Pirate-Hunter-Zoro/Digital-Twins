#!/bin/bash
#SBATCH --job-name=trd_prediction_pipeline
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/logs/trd_prediction_analysis_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/trd_prediction_analysis_err.txt

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

python -m scripts.digital_twins.embeddings.embedding_audit
python -m scripts.digital_twins.neighbors.llm_similarity_audit
python -m scripts.digital_twins.predictions.analyze_trd_prediction
python -m scripts.digital_twins.predictions.classical_ml

# Copy all files from the RESULTS_DIR to the local results/ directory for easier access
LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"

# Cancel the job
bash slurm_jobs/pipeline/cancel_pipeline.sh