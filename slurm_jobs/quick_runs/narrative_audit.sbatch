#!/bin/bash
#SBATCH --job-name=narrative_audit
#SBATCH --partition=c3_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/quick_runs/logs/narrative_audit_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/narrative_audit_err.txt

# Dumps a small reproducible sample of deterministic patient narratives
# (3 TRD-positive + 3 TRD-negative, by sorted patient hash) for manual review /
# a supplement example. CPU-only: reads narrative text straight from
# embeddings.db, no LLM or vLLM server. Narratives are deterministic (identical
# across encoders), so any encoder's .env is fine; the primary
# Qwen-Qwen3-Embedding-8B keeps output in that encoder's results tree.

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

# Write example narratives to ${RESULTS_DIR}/narrative_audit/
python -m scripts.pipeline.neighbors.narrative_audit

# Mirror the audit output into the base local results/ tree, preserving the
# per-encoder / per-judge substructure that the rest of results/ uses.
LOCAL_AUDIT_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}/narrative_audit"
mkdir -p "${LOCAL_AUDIT_DIR}"
rsync -av --delete "${RESULTS_DIR}/narrative_audit/" "${LOCAL_AUDIT_DIR}/"
