#!/bin/bash
#SBATCH --job-name=causal_validate
#SBATCH --partition=c3
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH -o slurm_jobs/pipeline/causal/logs/causal_validate_out.txt
#SBATCH -e slurm_jobs/pipeline/causal/logs/causal_validate_err.txt

# NOTE: no --gres=gpu on purpose. This job is CPU-only (CausalForestDML /
# RandomForest nuisances run with n_jobs=-1 on the allocated cores); every
# partition on this cluster happens to carry an A40, but we do not request one.

# Exit immediately if command fails
set -e

# ---- Conda configuration ----
# The causal stack (econml/CausalForestDML) lives in its OWN env, NOT the
# embedder_pipeline env the prediction jobs use.
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/causal_forest_env

# ---- Boilerplate env/bootstrap ----
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"

# Conda often dislikes 'set -u'—use this safe pattern:
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

# Load .env (SEED, ARTIFACTS_DIR, FEATURE_DATAFRAME_PATH, MDD_MED_DATE_CSV_PATH, ...)
ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ERROR: ${ENV_FILE} not found." >&2
  exit 11
fi
set -a; source "${ENV_FILE}"; set +a

python -m scripts.pipeline.causal.validate

# Copy all causal artifacts from ARTIFACTS_DIR to the local results/ directory for easier access.
# The causal arm never touches embeddings, so this is a flat mirror (no embedder/VLLM nesting).
LOCAL_RESULTS_DIR="results/causal_pipeline"
mkdir -p "${LOCAL_RESULTS_DIR}"
rsync -av "${ARTIFACTS_DIR}/causal_pipeline/" "${LOCAL_RESULTS_DIR}/"
