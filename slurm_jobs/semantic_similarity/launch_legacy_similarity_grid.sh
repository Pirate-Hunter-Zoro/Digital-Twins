#!/bin/bash

# The only job of this script is to LAUNCH other jobs!
set -e

# --- THIS IS THE MAGIC PART! ---
# This line finds the directory where THIS script lives, no matter where you run it from!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Now we build a perfect, absolute path to our template!
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/legacy_similarity_template.ssub"

# A quick check to make sure our magic worked!
if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find my beautiful submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the Games Begin! ---
for model_name in "${!MODELS[@]}"; do
    model_path=${MODELS[$model_name]}
    
    echo "🚀 Submitting job for ${model_name} from my cozy home at ${SCRIPT_DIR}!"
    # The sbatch command sends our worker bot off to the queue!
    sbatch "$SBATCH_TEMPLATE_PATH" "$model_name" "$model_path"
done

echo "🎉 All legacy jobs have been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on them!"