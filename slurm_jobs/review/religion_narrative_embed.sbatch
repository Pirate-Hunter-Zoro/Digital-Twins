#!/bin/bash
#SBATCH --job-name=religion_embed
#SBATCH --partition=c3_accel
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_jobs/review/logs/religion_embed_out.txt
#SBATCH -e slurm_jobs/review/logs/religion_embed_err.txt

# =============================================================================
# Review round of 2026-09-02, religion sensitivity item -- the GPU half.
#
# Two stages, and only two. The classifier fit is a SEPARATE job
# (religion_narrative_fit.sbatch), which is the lesson the parity round paid for:
# both of its narrative arms were killed at the wall part-way through the
# embedded logistic-regression grid search with everything upstream complete on
# disk, because the fit was sharing an allocation sized for the embedder and
# ordered behind cheaper work. Splitting them also lets the fit ask for 48 CPUs,
# which is what the published run used to fit the same grid on the same matrix.
#
#   render   42,579 narratives with the Religion field omitted, CPU
#   embed    the same encoder as the published run, GPU, ~3h
#
# THERE IS NO CONTROL ARM HERE AND THAT IS DELIBERATE. The parity round already
# rendered and embedded narrative_control -- a re-render with no content change --
# and the fit job scores this arm against it. That contrast is attributable to
# the removed field rather than to the rebuild, and it costs nothing because the
# control is already on disk.
# =============================================================================

set -e

CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline

module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

cd "${SLURM_SUBMIT_DIR}"

# The parity control arm has to exist, because it is what this arm is scored
# against. Fail now rather than after four hours of GPU.
CONTROL_DB="${ARTIFACTS_DIR}/review/parity/narrative_control/${EMBEDDER_MODEL_NAME}/embeddings.db"
if [[ ! -f "${CONTROL_DB}" ]]; then
  echo "ERROR: ${CONTROL_DB} not found; the parity control arm is the comparator." >&2
  exit 12
fi

ARM_DIR="${ARTIFACTS_DIR}/review/religion/narrative_no_religion"
mkdir -p "${ARM_DIR}"

# Every path the pipeline stages read or write is redirected into this arm, so no
# published artifact is touched: not the narratives, not the embedding database,
# not RESULTS_DIR.
export NARRATIVE_DROP_RELIGION=1
export NARRATIVES_DIR="${ARM_DIR}/narratives"
export EMBEDDINGS_DIR="${ARM_DIR}/${EMBEDDER_MODEL_NAME}"
export RESULTS_DIR="${ARM_DIR}"
export SCRUB_EMBEDDINGS=0
mkdir -p "${NARRATIVES_DIR}" "${EMBEDDINGS_DIR}"

echo "=== Stage 1/2: render religion-free narratives ==="
python -m scripts.pipeline.review.religion.render_narratives

echo "=== Stage 2/2: embed narratives ==="
python -m scripts.pipeline.embeddings.forge_embeddings

echo "Done: narratives rendered and embedded. Fit runs in religion_narrative_fit.sbatch."
