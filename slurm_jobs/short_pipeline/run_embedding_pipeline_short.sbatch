#!/bin/bash
#SBATCH --job-name=short_embedding_pipeline
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/short_pipeline/logs/embedding_pipeline_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/embedding_pipeline_err.txt
#SBATCH --gres=gpu:1

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

# Baseline pass: rebuild narratives + feature vectors + the BASELINE embeddings,
# reading the shared, model-independent sliced patient JSONs. forge_narratives
# also writes every per-spec ABLATED narrative dir (CPU-only), so the ablation
# embedding array downstream has narratives to read. The GPU-heavy ablation
# EMBEDDING itself is split out into run_ablation_embedding_array_short.sbatch
# (one spec per array task) so no single job embeds the whole cohort more than
# once -- that repeated full-cohort embed is what blew the c3_short wall before.
python -m scripts.pipeline.narratives.forge_narratives
python -m scripts.pipeline.vectors.forge_vectors
python -m scripts.pipeline.embeddings.forge_embeddings
