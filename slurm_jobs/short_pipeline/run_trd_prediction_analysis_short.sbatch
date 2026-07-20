#!/bin/bash
#SBATCH --job-name=short_trd_prediction_analysis
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/short_pipeline/logs/trd_prediction_analysis_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/trd_prediction_analysis_err.txt

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
N_THREADS="${SLURM_CPUS_PER_TASK}"

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# ---- Config: real .env, then the short-pipeline overrides ----
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

python -m scripts.pipeline.embeddings.embedding_audit

# LLM similarity audit only applies when judging ran (it did not here).
if [[ "$COMPUTE_LLM_SIMILARITY" == 1 ]]; then
  python -m scripts.pipeline.neighbors.llm_similarity_audit
fi

python -m scripts.pipeline.predictions.analyze_trd_prediction
python -m scripts.pipeline.predictions.classical_ml
python -m scripts.pipeline.predictions.feature_importance

# The ablation report consumes per-spec ablated embeddings; only run it when the
# ablation embeddings were actually built (see RUN_ABLATIONS in short_overrides.env).
if [[ "${RUN_ABLATIONS}" == 1 ]]; then
  python -m scripts.pipeline.predictions.ablation_runner
fi

# Mirror results into the repo under the normal results/<encoder>/<judge> path,
# same as the main pipeline. The 4B encoder name keeps this distinct from the
# live 8B run's mirror, so it is a first-class result, not a sandbox copy.
LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"

# Cancel the (already-finished) orchestrator/self entries.
bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh
