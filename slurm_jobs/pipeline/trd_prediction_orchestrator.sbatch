#!/bin/bash
#SBATCH --job-name=trd_prediction_orchestrator
#SBATCH --partition=c3_accel
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH -o slurm_jobs/pipeline/logs/trd_prediction_orchestrator_out.txt
#SBATCH -e slurm_jobs/pipeline/logs/trd_prediction_orchestrator_err.txt

ENV_FILE="${SLURM_SUBMIT_DIR}/.env"

CANCEL_SCRIPT="${SLURM_SUBMIT_DIR}/slurm_jobs/pipeline/cancel_pipeline.sh"
echo "#!/bin/bash" > "${CANCEL_SCRIPT}"
echo "echo 'Killing prediction pipeline...'" >> "${CANCEL_SCRIPT}"
chmod +x "${CANCEL_SCRIPT}"
echo "scancel ${SLURM_JOB_ID} # Orchestrator Job" >> "${CANCEL_SCRIPT}"

JSON_JOB_ID=$(sbatch --parsable slurm_jobs/pipeline/load_patient_json.sbatch)
echo "scancel ${JSON_JOB_ID} # Json Job" >> "${CANCEL_SCRIPT}"

EMBEDDING_JOB_ID=$(sbatch --parsable --dependency=afterok:$JSON_JOB_ID slurm_jobs/pipeline/run_embedding_pipeline.sbatch)
echo "scancel ${EMBEDDING_JOB_ID} # Embedding Job" >> "${CANCEL_SCRIPT}"

echo "Submitting the vLLM server job to begin once the embedding job is done..."
VLLM_JOB_ID=$(sbatch --parsable --dependency=afterok:$EMBEDDING_JOB_ID slurm_jobs/pipeline/start_vllm_server.sbatch)
echo "scancel ${VLLM_JOB_ID} # vLLM Server" >> "${CANCEL_SCRIPT}"

# We need to ensure the server starts AND updates the file before we try to source it.
echo "Waiting for vLLM server to become reachable..."

# Wait for the server to become reachable.
while true; do
  
  # See if the VLLM job is running yet
  STATUS=$(squeue -j $VLLM_JOB_ID -h -o "%t")

  if [ "$STATUS" == "R" ]; then
    echo "Server has started..."
    break
  fi
  if [[ "$STATUS" != "PD" && "$STATUS" != "CF" ]]; then
    echo "ERROR: vllm server failed to start!" >&2 # Print message to standard error
    exit 1 
  fi

  sleep 60
done

while true; do
  
  # Source the file to load the VLLM_URL into our environment.
  set -a; source "${ENV_FILE}"; set +a

  if curl -sf "${VLLM_URL}/health" >/dev/null; then
    # Server is live. Break the loop.
    echo "--> Server is live and responding at ${VLLM_URL}."
    break
  else
    echo "    Server not ready at ${VLLM_URL}. Retrying in 10 seconds..."
  fi
  
  sleep 10
done

echo "Launching the prediction pipeline..."
PREDICTION_JOB_ID=$(sbatch --parsable slurm_jobs/pipeline/run_trd_prediction_pipeline.sbatch)
echo "scancel ${PREDICTION_JOB_ID} # Prediction Pipeline" >> "${CANCEL_SCRIPT}"

echo "Launching evaluation pipeline to run once prediction is complete..."
ANALYSIS_JOB_ID=$(sbatch --parsable --dependency=afterok:$PREDICTION_JOB_ID slurm_jobs/pipeline/run_trd_prediction_analysis.sbatch)
echo "scancel ${ANALYSIS_JOB_ID} # Analysis Pipeline" >> "${CANCEL_SCRIPT}"

# Add self to kill list
echo "scancel ${SLURM_JOB_ID} # Orchestrator (Self)" >> "${CANCEL_SCRIPT}"

echo "Done."