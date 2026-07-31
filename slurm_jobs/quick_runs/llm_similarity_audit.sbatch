#!/bin/bash
#SBATCH --job-name=llm_similarity_audit
#SBATCH --partition=c3_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/quick_runs/logs/llm_similarity_audit_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/llm_similarity_audit_err.txt

# Regenerates the LLM-judge extremes audit (5 lowest + 5 highest overall_score
# anchor-neighbor pairs, each with both narratives + the judge JSON) for the
# manuscript Supplement S1.4 worked examples. CPU-only: it reads the canonical
# judgements.db and embeddings.db, no vLLM server required
# (Scorer(require_client=False)). Run with .env pointing EMBEDDER_MODEL_NAME at
# the primary Qwen-Qwen3-Embedding-8B encoder so the output lands in that
# encoder's results tree (narratives are deterministic, so the extremes
# themselves are embedder-independent).

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

# Extract the extreme-scored judgement reports to ${RESULTS_DIR}/llm_audit/
python -m scripts.pipeline.neighbors.llm_similarity_audit

# Mirror the audit output into the base local results/ tree, preserving the
# per-encoder / per-judge substructure that the rest of results/ uses.
LOCAL_AUDIT_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}/llm_audit"
mkdir -p "${LOCAL_AUDIT_DIR}"
rsync -av --delete "${RESULTS_DIR}/llm_audit/" "${LOCAL_AUDIT_DIR}/"
