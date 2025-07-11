#!/bin/bash

# The only job of this script is to LAUNCH the purity test job!
set -e

# --- THIS IS THE MAGIC PART! ---
# This line finds the directory where THIS script lives!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Now we build a perfect, absolute path to our template!
SBATCH_TEMPLATE_PATH="${SCRIPT_DIR}/purity_test_template.ssub"

# --- Our Champion! ---
# We can easily change this later if we want to test a different model!
CHAMPION_MODEL_TO_TEST="allenai/scibert_scivocab_uncased"

# A quick check to make sure our template is where we think it is!
if [ ! -f "$SBATCH_TEMPLATE_PATH" ]; then
    echo "ERROR! I can't find my beautiful submission template at ${SBATCH_TEMPLATE_PATH}"
    exit 1
fi

# --- Let the Test Begin! ---
echo "🚀 Submitting Category Purity Test for ${CHAMPION_MODEL_TO_TEST}..."
sbatch "$SBATCH_TEMPLATE_PATH" "$CHAMPION_MODEL_TO_TEST"

echo "🎉 The job has been submitted to the Slurm queue! Use 'squeue -u mferguson' to check on it!"