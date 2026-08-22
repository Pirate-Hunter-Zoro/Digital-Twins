#!/bin/bash
#SBATCH --job-name=ablation_backfill_score
#SBATCH --partition=c3_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -o slurm_jobs/quick_runs/logs/ablation_backfill_score_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/ablation_backfill_score_err.txt

# =============================================================================
# Stage 2 of 2: score every ablation spec against the frozen baseline and rewrite
# the ablation summary and figures.
#
# No GPU and no refitting. ablation_runner loads the cached best-estimator
# searchers from RESULTS_DIR/trained_models/ and applies them unchanged to each
# spec's embeddings, which is the whole point of the ablation design -- the
# question is what the perturbed INPUT does to a fixed model, not what a model
# retrained on perturbed input could recover.
#
# It rescores every spec on the slate, not only the backfilled one, because the
# summary table and the delta figures are written whole. That is deterministic
# and reproduces the previously published rows: the per-spec bootstrap seed is
# drawn from the spec's index in the slate, and the new spec was appended, so
# nothing already published shifts. (This is the reason ablation_registry says
# APPEND ONLY.)
#
# Prereqs: stage 1 finished, so every spec on the slate has an embeddings.db, and
# a completed classical_ml run, so trained_models/ and the baseline metrics JSON
# exist. Submit chained:
#
#   EMBED=$(sbatch --parsable slurm_jobs/quick_runs/ablation_backfill_embed.sbatch)
#   sbatch --dependency=afterok:$EMBED slurm_jobs/quick_runs/ablation_backfill_score.sbatch
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

# Load env variables
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

# Fail here rather than three hours in: the runner needs one cached searcher per
# classifier, and a missing one means the baseline run never completed.
for MODEL in logistic_regression random_forest gradient_boosting xgboost; do
  if [[ ! -f "${RESULTS_DIR}/trained_models/${MODEL}_EMBEDDED.joblib" ]]; then
    echo "ERROR: ${RESULTS_DIR}/trained_models/${MODEL}_EMBEDDED.joblib not found." >&2
    echo "       Run scripts.pipeline.predictions.classical_ml before scoring ablations." >&2
    exit 12
  fi
done

# Every spec on the slate must have been embedded, or the runner will score an
# empty directory and emit a delta against nothing.
MISSING=$(python - <<'PY'
import os
from pathlib import Path
from scripts.data_loading.ablation_registry import ABLATIONS
base = Path(os.environ['EMBEDDINGS_DIR'])
missing = [s["id"] for s in ABLATIONS if not (base / s["id"] / "embeddings.db").exists()]
print(",".join(missing))
PY
)
if [[ -n "${MISSING}" ]]; then
  echo "ERROR: no embeddings for ablation spec(s): ${MISSING}" >&2
  echo "       Run ablation_backfill_embed.sbatch for them first." >&2
  exit 13
fi

python -m scripts.pipeline.predictions.ablation_runner

# Mirror the rewritten summary and figures into the repo-local results/ tree,
# which is what the manuscript build reads.
LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/ablation_summary.csv" "${LOCAL_RESULTS_DIR}/"
rsync -av --include='ablation_*.png' --include='ROC_ablated_vs_baseline_*.png' \
  --exclude='*' "${RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"
