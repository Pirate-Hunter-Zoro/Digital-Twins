#!/bin/bash
#SBATCH --job-name=short_ablation_embedding
#SBATCH --partition=c3_short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/short_pipeline/logs/ablation_embedding_%a_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/ablation_embedding_%a_err.txt
#SBATCH --gres=gpu:1

# =============================================================================
# SHORT pipeline: per-spec ablation EMBEDDING, one spec per array task.
#
# The orchestrator submits this with --array=0-<N-1>%1, where N is the number of
# specs in ablation_registry.ABLATIONS. The %1 throttle serializes the tasks so
# only ONE GPU is ever held at a time; each task embeds a single ablation spec
# (one full-cohort pass, ~3h) and finishes well inside the c3_short wall. If more
# GPUs are free later, raising %1 -> %N parallelizes for free with no other edit.
#
# Prereqs (guaranteed by the afterok:<baseline embed job> dependency): the
# ablated narrative dirs (NARRATIVES_DIR/<spec_id>/) and the length CSV
# (narrative_pre_anchor_history_days.csv) that forge_narratives writes in the
# baseline job. Each spec writes its own EMBEDDINGS_DIR/<spec_id>/embeddings.db,
# consumed downstream by ablation_runner in the analysis job.
#
# c3_short enforces MaxMemPerCPU=12000 MB; 128G / 16 CPUs = 8 GB/CPU is legal.
# =============================================================================

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

# ---- Config: real .env, then the short-pipeline overrides ----
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

# forge_ablated_embeddings reads SLURM_ARRAY_TASK_ID and embeds ONLY that spec.
echo "Embedding ablation spec index ${SLURM_ARRAY_TASK_ID}..."
python -m scripts.pipeline.embeddings.forge_ablated_embeddings
