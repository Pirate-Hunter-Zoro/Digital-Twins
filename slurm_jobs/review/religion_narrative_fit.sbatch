#!/bin/bash
#SBATCH --job-name=religion_fit
#SBATCH --partition=c3_short
#SBATCH --time=08:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH -o slurm_jobs/review/logs/religion_fit_out.txt
#SBATCH -e slurm_jobs/review/logs/religion_fit_err.txt

# =============================================================================
# Religion sensitivity, narrative half -- CLASSIFIER FIT ONLY.
#
# Submit after religion_narrative_embed.sbatch, with a dependency:
#   EMB=$(sbatch --parsable slurm_jobs/review/religion_narrative_embed.sbatch)
#   sbatch --dependency=afterok:${EMB} slurm_jobs/review/religion_narrative_fit.sbatch
#
# Forty-eight CPUs and no GPU, for the reasons recorded in
# parity_narrative_fit.sbatch: GridSearchCV runs 16 worker processes and the
# published run fit this grid on a 4,096-column matrix in 4.9h with 48 CPUs and
# BLAS left threaded. Do NOT pin OMP_NUM_THREADS here -- the feature arm pins it
# because 91 columns make BLAS threading pure overhead, and 4,096 do not.
#
# Two contrasts come out of it: against the parity round's narrative_control,
# which is the field-attributable one, and against the published embedded arm,
# which carries the re-render inside it as well.
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

ARM_DIR="${ARTIFACTS_DIR}/review/religion/narrative_no_religion"
export EMBEDDINGS_DIR="${ARM_DIR}/${EMBEDDER_MODEL_NAME}"

# Preflight. load_embedded_matrix raises on a short database, but it raises after
# the split and the load; and a missing control arm would only surface later
# still, after the fit has already been paid for.
EMB_DB="${EMBEDDINGS_DIR}/embeddings.db"
if [[ ! -f "${EMB_DB}" ]]; then
  echo "ERROR: ${EMB_DB} not found; the embed job never finished." >&2
  exit 12
fi
CONTROL_PREDS="${ARTIFACTS_DIR}/review/parity/narrative_control/test_predictions.parquet"
if [[ ! -f "${CONTROL_PREDS}" ]]; then
  echo "ERROR: ${CONTROL_PREDS} not found; the field-attributable contrast needs it." >&2
  exit 13
fi

# RESULTS_DIR points at the PUBLISHED results, which is where the published
# test_predictions parquet the comparison is paired against lives.
RESULTS_DIR="${ARTIFACTS_DIR}/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}" \
  python -m scripts.pipeline.review.religion.run_embedded_arm
python -m scripts.pipeline.review.religion.summarize

LOCAL_RESULTS_DIR="results/review/religion"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av --exclude '*.joblib' --exclude 'narratives/' --exclude '*.db' \
  "${ARTIFACTS_DIR}/review/religion/" "${LOCAL_RESULTS_DIR}/"

echo "Done: religion narrative fit."
