#!/bin/bash

ENV_FILE=".env"

echo "Submitting the vLLM server job..."
sbatch slurm_jobs/digital_twins/start_vllm_server.sbatch

# We need to ensure the server starts AND updates the file before we try to source it.
echo "Waiting for vLLM server to become reachable..."

while true; do
  
  # 1. Source the file to load the VLLM_URL into our environment.
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
JOB_ID=$(sbatch --parsable slurm_jobs/digital_twins/run_trd_prediction_pipeline.sbatch)

echo "Launching evaluation pipeline to run once prediction is complete..."
sbatch --dependency=afterok:$JOB_ID slurm_jobs/digital_twins/run_trd_prediction_analysis.sbatch

echo "Done."