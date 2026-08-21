#!/bin/bash
#SBATCH --job-name=plot_manuscript_figures
#SBATCH --partition=c3_short
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -o slurm_jobs/quick_runs/logs/plot_manuscript_figures_out.txt
#SBATCH -e slurm_jobs/quick_runs/logs/plot_manuscript_figures_err.txt

# Regenerates the presentation-layer figures from artifacts already on disk:
#   - the two-panel discrimination figure: per-model AUCs with marginal intervals,
#     above the paired EMBEDDED-minus-FEATURE differences
#   - the four per-classifier feature-importance panels
# Neither step refits a model, so this is safe and cheap to re-run whenever only
# figure styling changes. It needs a completed classical_ml + feature_importance
# run to read from; it will fail loudly rather than recompute if either is absent.
# The paired panel additionally needs classical_ml's test_predictions_*.parquet
# tables, so a results tree predating those will fail on the missing file rather
# than quietly drop the panel.

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

python -m scripts.pipeline.predictions.plot_discrimination_forest
python -m scripts.pipeline.predictions.redraw_feature_importance

# Mirror the regenerated figures into the repo-local results/ tree
LOCAL_RESULTS_DIR="results/${EMBEDDER_MODEL_NAME}/${VLLM_MODEL_NAME}"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${RESULTS_DIR}/discrimination_forest_EMBEDDED_vs_FEATURE.png" "${LOCAL_RESULTS_DIR}/"
rsync -av "${RESULTS_DIR}/discrimination_paired_deltas.json" "${LOCAL_RESULTS_DIR}/"
rsync -av "${RESULTS_DIR}/feature_importance/" "${LOCAL_RESULTS_DIR}/feature_importance/"
