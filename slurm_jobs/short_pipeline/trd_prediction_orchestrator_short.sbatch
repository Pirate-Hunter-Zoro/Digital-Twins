#!/bin/bash
#SBATCH --job-name=trd_short_orchestrator
#SBATCH --partition=c3_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/short_pipeline/logs/orchestrator_out.txt
#SBATCH -e slurm_jobs/short_pipeline/logs/orchestrator_err.txt

# =============================================================================
# SHORT pipeline orchestrator (c3_short, LLM judging OFF).
#
# DAG:
#
#     baseline_embed ─┬─> prediction (array 0-0) ──────────┐
#                     │                                     ├─> analysis
#                     └─> ablation_embed (array 0-N%1) ─────┘
#
#   * baseline_embed : narratives + feature vectors + BASELINE embed. Its
#     forge_narratives also writes every per-spec ablated narrative dir (CPU).
#   * prediction     : baseline neighborhoods, CPU only, needs the baseline
#     embed. Runs in PARALLEL with the ablation array.
#   * ablation_embed : one spec per task, %1 => one GPU at a time, each task one
#     full-cohort pass that clears the wall. Needs the ablated narrative dirs
#     from baseline_embed. Skipped entirely when RUN_ABLATIONS != 1.
#   * analysis       : waits on BOTH prediction and (if run) the whole ablation
#     array; only then does ablation_runner have every per-spec embeddings.db.
#
# The JSON build is SKIPPED on purpose: the sliced patient JSONs are model-
# independent and already exist under ANALYSIS_DIR from the main run, and
# rebuilding them is what hit the Isilon quota before. Outputs land in the shared
# artifacts/ tree keyed by the 4B EMBEDDER_MODEL_NAME (see short_overrides.env),
# so the live 8B run is never touched.
#
# LLM judging is intentionally unsupported here (it cannot clear the wall).
# =============================================================================

# ---- Conda (needed to read the ablation-spec count from the registry) ----
CONDA_ENV=/media/studies/ehr_study/analysis/mferguson/venvs/embedder_pipeline
module purge
module load Anaconda3/2025.06-0
eval "$(conda shell.bash hook)"
set +u; conda activate "${CONDA_ENV}"; set -u || exit 10

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
set -a; source "${ENV_FILE}"; set +a
OVERRIDES_FILE="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/short_overrides.env"
set -a; source "${OVERRIDES_FILE}"; set +a

cd "${SLURM_SUBMIT_DIR}"

CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/short_pipeline/cancel_short_pipeline.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing SHORT prediction pipeline...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"
echo "scancel ${SLURM_JOB_ID} # Orchestrator Job" >> "${CANCEL_SCRIPT}"

# 1) Baseline: narratives + vectors + baseline embed (single GPU, one pass).
echo "Launching the baseline embedding job (narratives + vectors + baseline embed)..."
EMBEDDING_JOB_ID=$(sbatch --parsable slurm_jobs/short_pipeline/run_embedding_pipeline_short.sbatch)
echo "scancel ${EMBEDDING_JOB_ID} # Baseline Embedding Job" >> "${CANCEL_SCRIPT}"

# 2) Prediction: baseline neighborhoods (CPU, no vLLM), needs only the baseline embed.
echo "Launching the prediction pipeline (array 0-0, no vLLM)..."
PREDICTION_JOB_ID=$(sbatch --parsable --array=0-0 --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/short_pipeline/run_trd_prediction_pipeline_short.sbatch)
echo "scancel ${PREDICTION_JOB_ID} # Prediction Pipeline" >> "${CANCEL_SCRIPT}"

# 3) Ablations: one spec per array task, serialized on one GPU (%1). Array width
#    is derived from the registry so adding/removing a spec needs no edit here.
if [[ "${RUN_ABLATIONS}" == 1 ]]; then
  N_SPECS=$(python -c "from scripts.data_loading.ablation_registry import ABLATIONS; print(len(ABLATIONS))")
  if [[ -z "${N_SPECS}" || "${N_SPECS}" -lt 1 ]]; then
    echo "ERROR: could not read ablation spec count (got '${N_SPECS}')." >&2
    exit 20
  fi
  LAST_IDX=$((N_SPECS - 1))
  echo "Launching the ablation embedding array (0-${LAST_IDX}%1, one spec per task, one GPU at a time)..."
  ABLATION_JOB_ID=$(sbatch --parsable --array=0-${LAST_IDX}%1 --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/short_pipeline/run_ablation_embedding_array_short.sbatch)
  echo "scancel ${ABLATION_JOB_ID} # Ablation Embedding Array" >> "${CANCEL_SCRIPT}"
fi

# 4) Neighbor analysis (light): embedding_audit + analyze_trd_prediction. Needs
#    the baseline neighborhoods, so it depends on prediction ONLY -- not on the
#    ablation array (it never reads the per-spec embeddings). Split from the
#    classical-ML arm 2026-07-22 so the ~2.5h GridSearchCV no longer shares its
#    wall.
echo "Launching the neighbor-analysis job (dependency: afterok:$PREDICTION_JOB_ID)..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$PREDICTION_JOB_ID slurm_jobs/short_pipeline/run_trd_prediction_analysis_short.sbatch)
echo "scancel ${ANALYSIS_JOB_ID} # Neighbor Analysis" >> "${CANCEL_SCRIPT}"

# 5) Classical ML (the grid-search wall-hitter) + feature importance + ablation
#    report, in its own dedicated 9h job. Independent of the neighbor arm, so it
#    runs in PARALLEL with prediction/analysis. Needs the baseline embed, plus
#    every per-spec ablation embeddings.db (for ablation_runner) when
#    RUN_ABLATIONS=1.
ML_DEP="afterok:$EMBEDDING_JOB_ID"
if [[ "${RUN_ABLATIONS}" == 1 ]]; then
  ML_DEP="${ML_DEP}:$ABLATION_JOB_ID"
fi
echo "Launching the classical-ML job (dependency: ${ML_DEP})..."
ML_JOB_ID=$(sbatch --parsable --dependency=${ML_DEP} slurm_jobs/short_pipeline/run_classical_ml_short.sbatch)
echo "scancel ${ML_JOB_ID} # Classical ML" >> "${CANCEL_SCRIPT}"

# 6) Mirror: the SINGLE results rsync into the repo, after BOTH arms succeed, so
#    it copies a complete result set exactly once.
echo "Launching the results-mirror job (dependency: afterok:${ANALYSIS_JOB_ID}:${ML_JOB_ID})..."
MIRROR_JOB_ID=$(sbatch --parsable --dependency=afterok:${ANALYSIS_JOB_ID}:${ML_JOB_ID} slurm_jobs/short_pipeline/run_results_mirror_short.sbatch)
echo "scancel ${MIRROR_JOB_ID} # Results Mirror" >> "${CANCEL_SCRIPT}"

echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"
echo "Done. Cancel the whole chain with: bash slurm_jobs/short_pipeline/cancel_short_pipeline.sh"
