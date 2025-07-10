#!/bin/bash

#SBATCH --job-name=legacy_sim_grid
#SBATCH --partition=c3_accel
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G

# Activate your wonderful Conda environment!
source /opt/apps/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate hugging_env

set -e

# The path to our project, because we're organized!
PROJECT_ROOT="/home/librad.laureateinstitute.org/mferguson/Digital-Twins"
# Our new legacy script!
PYTHON_SCRIPT="$PROJECT_ROOT/scripts/semantic_similarity/embed_term_pairs_legacy.py"
# The term pairs we want to test!
TERM_PAIRS_FILE="$PROJECT_ROOT/data/term_pairs.csv"
# Where the results will go!
OUTPUT_DIR="$PROJECT_ROOT/results/semantic_similarity_legacy"

mkdir -p "$OUTPUT_DIR"

# --- Our Legacy Contestants! ---
declare -A MODELS
MODELS["glove-6B-300d"]="/media/scratch/mferguson/legacy_models/glove.6B.300d.txt"
MODELS["word2vec-google-news-300"]="/media/scratch/mferguson/legacy_models/GoogleNews-vectors-negative300.bin"

# --- Let the Games Begin! ---
for model_name in "${!MODELS[@]}"; do
    model_path=${MODELS[$model_name]}
    output_file="$OUTPUT_DIR/similarity_${model_name}.csv"
    
    echo "🚀 Launching job for ${model_name}!"

    CMD="python $PYTHON_SCRIPT --model_path \"$model_path\" --term_pairs_file \"$TERM_PAIRS_FILE\" --output_file \"$output_file\""
    
    # We have to add a special flag for our Word2Vec friend!
    if [[ "$model_name" == "word2vec-google-news-300" ]]; then
        CMD="$CMD --is_word2vec"
    fi

    # Run the command!
    eval $CMD
done

echo "🎉 All legacy jobs have been launched! Let's get that data!"