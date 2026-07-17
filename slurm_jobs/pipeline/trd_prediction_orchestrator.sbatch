#!/bin/bash
#SBATCH --job-name=trd_prediction_orchestrator
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/pipeline/logs/trd_prediction_orchestrator_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/trd_prediction_orchestrator_err.txt

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"
# Source the file for NUM_VLLM_SERVERS, ENDPOINTS_DIR, COMPUTE_LLM_SIMILARITY, and other flags.
set -a; source "${ENV_FILE}"; set +a

CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/pipeline/cancel_pipeline.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing prediction pipeline...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"
echo "scancel ${SLURM_JOB_ID} # Orchestrator Job" >> "${CANCEL_SCRIPT}"

JSON_JOB_ID=$(sbatch --parsable slurm_jobs/pipeline/load_patient_json.sbatch)
echo "scancel ${JSON_JOB_ID} # Json Job" >> "${CANCEL_SCRIPT}"

EMBEDDING_JOB_ID=$(sbatch --parsable --dependency=afterok:$JSON_JOB_ID slurm_jobs/pipeline/run_embedding_pipeline.sbatch)
echo "scancel ${EMBEDDING_JOB_ID} # Embedding Job" >> "${CANCEL_SCRIPT}"

# Number of vLLM servers to launch (and matching pipeline shards). Default 1.
NUM_VLLM_SERVERS="${NUM_VLLM_SERVERS:-1}"
LAST_IDX=$(( NUM_VLLM_SERVERS - 1 ))

# Without LLM judging there is nothing to shard: a single pipeline task.
ARRAY_SPEC="0-0"

if [[ "$COMPUTE_LLM_SIMILARITY" == 1 ]]; then

  # Fan the pipeline out to one task per server.
  ARRAY_SPEC="0-${LAST_IDX}"

  # Wipe stale endpoint files so a task never reads a URL from a prior run.
  mkdir -p "${ENDPOINTS_DIR}"
  rm -f "${ENDPOINTS_DIR}"/endpoint_*.url

  echo "Submitting ${NUM_VLLM_SERVERS} vLLM server(s) to begin once the embedding job is done..."
  VLLM_JOB_ID=$(sbatch --parsable --array="0-${LAST_IDX}" --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/pipeline/start_vllm_server.sbatch)
  echo "scancel ${VLLM_JOB_ID} # vLLM Server Array" >> "${CANCEL_SCRIPT}"

  # Wait for every server to publish its endpoint AND answer /health.
  echo "Waiting for ${NUM_VLLM_SERVERS} vLLM server(s) to become reachable..."
  for IDX in $(seq 0 "${LAST_IDX}"); do
    EP_FILE="${ENDPOINTS_DIR}/endpoint_${IDX}.url"
    echo "  Waiting on server ${IDX} (${EP_FILE})..."
    while true; do

      # Bail out if this array task hit a terminal (failed) state.
      STATE=$(squeue -j "${VLLM_JOB_ID}_${IDX}" -h -o "%t" 2>/dev/null)
      if [[ "$STATE" =~ ^(F|CA|TO|NF|BF|OOM|DL|SE)$ ]]; then
        echo "ERROR: vLLM server ${IDX} entered terminal state ${STATE}!" >&2
        exit 1
      fi

      # Once the endpoint file is written, probe its health endpoint.
      if [[ -s "$EP_FILE" ]]; then
        URL=$(cat "$EP_FILE")
        if curl -sf "${URL}/health" >/dev/null; then
          echo "  --> Server ${IDX} live at ${URL}."
          break
        fi
      fi

      sleep 10
    done
  done

fi

echo "Launching the prediction pipeline (array ${ARRAY_SPEC})..."
PREDICTION_JOB_ID=$(sbatch --parsable --array="${ARRAY_SPEC}" --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/pipeline/run_trd_prediction_pipeline.sbatch)
echo "scancel ${PREDICTION_JOB_ID} # Prediction Pipeline" >> "${CANCEL_SCRIPT}"

# Analysis waits on the prediction array by default; the LLM path reinserts the merge below.
ANALYSIS_DEP=$PREDICTION_JOB_ID

if [[ "$COMPUTE_LLM_SIMILARITY" == 1 ]]; then

  VLLM_KILL_JOB_ID=$(
    sbatch --parsable --dependency=afterany:$PREDICTION_JOB_ID \
      --partition=c3_short \
      --time=00:01:00 \
      --wrap="scancel $VLLM_JOB_ID"
  )
  echo "scancel ${VLLM_KILL_JOB_ID} # vLLM Kill Stub" >> "${CANCEL_SCRIPT}"

  # Merge the per-shard judgements_{i}.db into the canonical judgements.db before analysis reads it.
  echo "Chaining judgements merge (afterok on the full prediction array)..."
  MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:$PREDICTION_JOB_ID slurm_jobs/pipeline/merge_judgements.sbatch)
  echo "scancel ${MERGE_JOB_ID} # Judgements Merge" >> "${CANCEL_SCRIPT}"
  ANALYSIS_DEP=$MERGE_JOB_ID

fi

echo "Launching evaluation pipeline to run once prediction is complete..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$ANALYSIS_DEP slurm_jobs/pipeline/run_trd_prediction_analysis.sbatch)
echo "scancel ${ANALYSIS_JOB_ID} # Analysis Pipeline" >> "${CANCEL_SCRIPT}"

# Add self to kill list
echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"

echo "Done."