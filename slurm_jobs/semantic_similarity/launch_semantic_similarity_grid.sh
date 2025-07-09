#!/bin/bash

# === Find the project root directory, no matter where the script is run from! ===
# This gets the directory where this script itself is located.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# This goes up two levels from the script's location to find the main project folder.
PROJECT_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))

# === Read Embedding Models from File ===
# Now we build the full, glorious path to our candidates file!
CANDIDATE_FILE="$PROJECT_ROOT/data/vectorizer_candidates.txt"
if [ ! -f "$CANDIDATE_FILE" ]; then
    echo "ERROR: Vectorizer candidate file not found at $CANDIDATE_FILE"
    exit 1
fi
mapfile -t VECTORIZERS < "$CANDIDATE_FILE"

TEMPLATE="slurm_jobs/semantic_similarity/semantic_similarity_template.ssub"

mkdir -p logs

echo "Found ${#VECTORIZERS[@]} models to test. Let's do this!"

for VEC in "${VECTORIZERS[@]}"; do
  # Sanitize model name for file paths
  JOB_VEC_NAME=$(echo "$VEC" | tr '/' '_')

  JOB_NAME="semantic_sim_${JOB_VEC_NAME}"
  LOG_OUT="logs/${JOB_NAME}_out.txt"
  LOG_ERR="logs/${JOB_NAME}_err.txt"

  echo "Launching $JOB_NAME..."

  sbatch \
    --job-name="$JOB_NAME" \
    --output="$LOG_OUT" \
    --error="$LOG_ERR" \
    "$TEMPLATE" "$VEC"

  sleep 1
done

echo "All jobs launched! Now we wait for the glorious data!"